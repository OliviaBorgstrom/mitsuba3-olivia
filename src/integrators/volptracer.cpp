#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-ptracer:

Particle tracer (:monosp:`ptracer`)
-----------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

 * - samples_per_pass
   - |bool|
   - If specified, divides the workload in successive passes with :paramtype:`samples_per_pass`
     samples per pixel.

This integrator traces rays starting from light sources and attempts to connect them
to the sensor at each bounce.
It does not support media (volumes).

Usually, this is a relatively useless rendering technique due to its high variance, but there
are some cases where it excels. In particular, it does a good job on scenes where most scattering
events are directly visible to the camera.

Note that unlike sensor-based integrators such as :ref:`path <integrator-path>`, it is not
possible to divide the workload in image-space tiles. The :paramtype:`samples_per_pass` parameter
allows splitting work in successive passes of the given sample count per pixel. It is particularly
useful in wavefront mode.

.. tabs::
    .. code-tab::  xml

        <integrator type="ptracer">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'ptracer',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
//Need to add to the AdjointIntegrator thing, or make a new one to pass in medium
class VolumetricParticleTracerIntegrator final : public MyIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MyIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_max_depth)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr, Medium, MediumPtr)

    VolumetricParticleTracerIntegrator(const Properties &props) : Base(props) { }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, dr::eq(idx, 1u)) = spec[1];
            dr::masked(m, dr::eq(idx, 2u)) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }

    void sample(const Scene *scene, const Sensor *sensor, Sampler *sampler,
                ImageBlock *block, ScalarFloat sample_scale, const Medium *initial_medium) const override { //,const Medium *initial_medium) const override {
        // Account for emitters directly visible from the sensor
        MediumPtr medium = initial_medium;

        if (m_max_depth != 0 && !m_hide_emitters)
            sample_visible_emitters(scene, sensor, sampler, block, sample_scale);

        // Primary & further bounces illumination
        auto [ray, throughput] = prepare_ray(scene, sensor, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = dr::neq(throughput_max, 0.f);

        trace_light_ray(ray, scene, sensor, sampler, throughput, block,
                        sample_scale, medium, active);
    }

    /**
     * Samples an emitter in the scene and connects it directly to the sensor,
     * splatting the emitted radiance to the given image block.
     */
    void sample_visible_emitters(const Scene *scene, const Sensor *sensor,
                                 Sampler *sampler, ImageBlock *block,
                                 ScalarFloat sample_scale) const {
        // 1. Time sampling
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d() * sensor->shutter_open_time();

        // 2. Emitter sampling (select one emitter)
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d());

        EmitterPtr emitter =
            dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);

        // Don't connect delta emitters with sensor (both position and direction)
        Mask active = !has_flag(emitter->flags(), EmitterFlags::Delta);

        // 3. Emitter position sampling
        Spectrum emitter_weight = dr::zeros<Spectrum>();
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        // 3.a. Infinite emitters
        Mask is_infinite = has_flag(emitter->flags(), EmitterFlags::Infinite),
             active_e = active && is_infinite;
        if (dr::any_or<true>(active_e)) {
            /* Sample a direction toward an envmap emitter starting
               from the center of the scene (the sensor is not part of the
               scene's bounding box, which could otherwise cause issues.) */
            Interaction3f ref_it(0.f, time, dr::zeros<Wavelength>(),
                                 sensor->world_transform().translation());

            auto [ds, dir_weight] = emitter->sample_direction(
                ref_it, sampler->next_2d(active), active_e);

            /* Note: `dir_weight` already includes the emitter radiance, but
               that will be accounted for again when sampling the wavelength
               below. Instead, we recompute just the factor due to the PDF.
               Also, convert to area measure. */
            emitter_weight[active_e] =
                dr::select(ds.pdf > 0.f, dr::rcp(ds.pdf), 0.f) *
                dr::sqr(ds.dist);

            si[active_e] = SurfaceInteraction3f(ds, ref_it.wavelengths);
        }

        // 3.b. Finite emitters
        active_e = active && !is_infinite;
        if (dr::any_or<true>(active_e)) {
            auto [ps, pos_weight] =
                emitter->sample_position(time, sampler->next_2d(active), active_e);

            emitter_weight[active_e] = pos_weight;
            si[active_e] = SurfaceInteraction3f(ps, dr::zeros<Wavelength>());
        }

        /* 4. Connect to the sensor.
           Query sensor for a direction connecting to `si.p`, which also
           produces UVs on the sensor (for splatting). The resulting direction
           points from si.p (on the emitter) toward the sensor. */
        auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, sampler->next_2d(), active);
        si.wi = sensor_ds.d;

        // 5. Sample spectrum of the emitter (accounts for its radiance)
        auto [wavelengths, wav_weight] =
            emitter->sample_wavelengths(si, sampler->next_1d(active), active);
        si.wavelengths = wavelengths;
        si.shape       = emitter->shape();

        Spectrum weight = emitter_idx_weight * emitter_weight * wav_weight * sensor_weight;

        // No BSDF passed (should not evaluate it since there's no scattering)
        connect_sensor(scene, si, sensor_ds, nullptr, weight, block, sample_scale, active);
    }

    /// Samples a ray from a random emitter in the scene.
    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler) const {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d() * sensor->shutter_open_time();

        // Prepare random samples.
        Float wavelength_sample  = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample);

        return { ray, ray_weight };
    }

    /**
     * Intersects the given ray with the scene and recursively trace using
     * BSDF sampling. The given `throughput` should account for emitted
     * radiance from the sampled light source, wavelengths sampling weights,
     * etc. At each interaction, we attempt to connect to the sensor and add
     * the current radiance to the given `block`.
     *
     * Note: this will *not* account for directly visible emitters, since
     * they require a direct connection from the emitter to the sensor. See
     * \ref sample_visible_emitters.
     *
     * \return The radiance along the ray and an alpha value.
     */
    std::pair<Spectrum, Float>
    trace_light_ray(Ray3f ray, const Scene *scene, const Sensor *sensor,
                    Sampler *sampler, Spectrum throughput, ImageBlock *block,
                    ScalarFloat sample_scale,  MediumPtr medium, Mask active = true) const {
        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Int32 depth = 1;

        Mask specular_chain = active && !m_hide_emitters;
        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) dr::array_size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }
        /* ---------------------- Path construction ------------------------- */
        // First intersection from the emitter to the scene
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        Mask needs_intersection = true;
        Interaction3f last_scatter_event = dr::zeros<Interaction3f>();
        Float last_scatter_direction_pdf = 1.f;

        active &= si.is_valid();
        if (m_max_depth >= 0)
            active &= depth < m_max_depth;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Particle Tracer Integrator", active, depth, ray,
                            throughput, si, eta, sampler,medium,needs_intersection,
                            mei, last_scatter_event, specular_chain, last_scatter_direction_pdf);

        // Incrementally build light path using BSDF sampling.
        while (loop(active)) {
            BSDFPtr bsdf = si.bsdf(ray);

            /* Connect to sensor and splat if successful. Sample a direction
               from the sensor to the current surface point. */
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, sampler->next_2d(), active);
            connect_sensor(scene, si, sensor_ds, bsdf,
                           throughput * sensor_weight, block, sample_scale,
                           active);
            
            /* ----------------------- Medium sampling ------------------------ */
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false,
                 escaped_medium = false;
            
            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering
            Mask is_spectral = active_medium;
            Mask not_spectral = false;
            if (dr::any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (dr::any_or<true>(active_medium)) {
                mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = mei.t;
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
                needs_intersection &= !active_medium;

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                if (dr::any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->transmittance_eval_pdf(mei, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(throughput, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                // Handle null and real scatter events
                Mask null_scatter = sampler->next_1d(active_medium) >= index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel);

                act_null_scatter |= null_scatter && active_medium;
                act_medium_scatter |= !act_null_scatter && active_medium;

                if (dr::any_or<true>(is_spectral && act_null_scatter))
                    dr::masked(throughput, is_spectral && act_null_scatter) *=
                        mei.sigma_n * index_spectrum(mei.combined_extinction, channel) /
                        index_spectrum(mei.sigma_n, channel);

                dr::masked(depth, act_medium_scatter) += 1;
                dr::masked(last_scatter_event, act_medium_scatter) = mei;
            }
            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
            
            if (dr::any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera = active_surface && dr::eq(depth, 0u);
                Mask count_direct = ray_from_camera || specular_chain;
                EmitterPtr emitter = si.emitter(scene);
                Mask active_e = active_surface && dr::neq(emitter, nullptr)
                                && !(dr::eq(depth, 0u) && m_hide_emitters);
                if (dr::any_or<true>(active_e)) {
                    Float emitter_pdf = 1.0f;
                    if (dr::any_or<true>(active_e && !count_direct)) {
                        // Get the PDF of sampling this emitter using next event estimation
                        DirectionSample3f ds(scene, si, last_scatter_event);
                        emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_e);
                    }
                    Spectrum emitted = emitter->eval(si, active_e);
                    Spectrum contrib = dr::select(count_direct, throughput * emitted,
                                                  throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted);
                    throughput += contrib;
                }
            }

            // making volptracer using env light
            // make volpath using photon. 
            active_surface &= si.is_valid();
            if (dr::any_or<true>(active_surface)) {
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);
                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

                if (likely(dr::any_or<true>(active_e))) {
                    Spectrum emitted(1.0f);
                    auto [ds, emitter_val] = scene->sample_emitter_direction(si, sampler->next_2d(active), false, active_e);
                    //auto [emitted, ds] = 1,1;
                    //sample_emitter(si, scene, sampler, medium, channel, active_e);

                    // Query the BSDF for that emitter-sampled direction
                    Vector3f wo       = si.to_local(ds.d);
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                    // Determine probability of having sampled that same
                    // direction using BSDF sampling.
                    Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);
                    throughput *= bsdf_val * mis_weight(ds.pdf, dr::select(ds.delta, 0.f, bsdf_pdf)) * emitted;
                }
            }

            /* ----------------------- BSDF sampling ------------------------ */
            // Sample BSDF * cos(theta).
            BSDFContext ctx(TransportMode::Importance);
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si, sampler->next_1d(active),
                             sampler->next_2d(active), active);

            // Using geometric normals (wo points to the camera)
            Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                  wo_dot_geo_n = dr::dot(si.n, si.to_world(bs.wo));

            // Prevent light leaks due to shading normals
            active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                      (wo_dot_geo_n * Frame3f::cos_theta(bs.wo) > 0.f);

            // Adjoint BSDF for shading normals -- [Veach, p. 155]
            Float correction = dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                       (Frame3f::cos_theta(bs.wo) * wi_dot_geo_n));
            throughput *= bsdf_val * correction;
            eta *= bs.eta;

            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            if (dr::none_or<false>(active))
                break;

            // Intersect the BSDF ray against scene geometry (next vertex).
            ray = si.spawn_ray(si.to_world(bs.wo));
            si = scene->ray_intersect(ray, active);

            depth++;
            if (m_max_depth >= 0)
                active &= depth < m_max_depth;
            active &= si.is_valid();

            // Russian Roulette
            Mask use_rr = depth > m_rr_depth;
            if (dr::any_or<true>(use_rr)) {
                Float q = dr::minimum(
                    dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), 0.95f);
                dr::masked(active, use_rr) &= sampler->next_1d(active) < q;
                dr::masked(throughput, use_rr) *= dr::rcp(q);
            }
        }

        return { throughput, 1.f };
        
    }

    /**
     * Attempt connecting the given point to the sensor.
     *
     * If the point to connect is on the surface (non-null `bsdf` values),
     * evaluate the BSDF in the direction of the sensor.
     *
     * Finally, splat `weight` (with all appropriate factors) to the
     * given image block.
     *
     * \return The quantity that was accumulated to the block.
     */
    Spectrum connect_sensor(const Scene *scene, const SurfaceInteraction3f &si,
                            const DirectionSample3f &sensor_ds,
                            const BSDFPtr &bsdf, const Spectrum &weight,
                            ImageBlock *block, ScalarFloat sample_scale,
                            Mask active) const {
        active &= (sensor_ds.pdf > 0.f) &&
                  dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
        if (dr::none_or<false>(active))
            return 0.f;

        // Check that sensor is visible from current position (shadow ray).
        Ray3f sensor_ray = si.spawn_ray_to(sensor_ds.p);
        active &= !scene->ray_test(sensor_ray, active);
        if (dr::none_or<false>(active))
            return 0.f;

        // Foreshortening term and BSDF value for that direction (for surface interactions).
        Spectrum result = 0.f;
        Spectrum surface_weight = 1.f;
        Vector3f local_d        = si.to_local(sensor_ray.d);
        Mask on_surface         = active && dr::neq(si.shape, nullptr);
        if (dr::any_or<true>(on_surface)) {
            /* Note that foreshortening is only missing for directly visible
               emitters associated with a shape. Otherwise it's included in the
               BSDF. Clamp negative cosines (zero value if behind the surface). */

            surface_weight[on_surface && dr::eq(bsdf, nullptr)] *=
                dr::maximum(0.f, Frame3f::cos_theta(local_d));

            on_surface &= dr::neq(bsdf, nullptr);
            if (dr::any_or<true>(on_surface)) {
                BSDFContext ctx(TransportMode::Importance);
                // Using geometric normals
                Float wi_dot_geo_n = dr::dot(si.n, si.to_world(si.wi)),
                      wo_dot_geo_n = dr::dot(si.n, sensor_ray.d);

                // Prevent light leaks due to shading normals
                Mask valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                             (wo_dot_geo_n * Frame3f::cos_theta(local_d) > 0.f);

                // Adjoint BSDF for shading normals -- [Veach, p. 155]
                Float correction = dr::select(valid,
                    dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                            (Frame3f::cos_theta(local_d) * wi_dot_geo_n)), 0.f);

                surface_weight[on_surface] *=
                    correction * bsdf->eval(ctx, si, local_d, on_surface);
            }
        }

        /* Even if the ray is not coming from a surface (no foreshortening),
           we still don't want light coming from behind the emitter. */
        Mask not_on_surface = active && dr::eq(si.shape, nullptr) && dr::eq(bsdf, nullptr);
        if (dr::any_or<true>(not_on_surface)) {
            Mask invalid_side = Frame3f::cos_theta(local_d) <= 0.f;
            surface_weight[not_on_surface && invalid_side] = 0.f;
        }

        result = weight * surface_weight * sample_scale;

        /* Splatting, adjusting UVs for sensor's crop window if needed.
           The crop window is already accounted for in the UV positions
           returned by the sensor, here we just need to compensate for
           the block's offset that will be applied in `put`. */
        Float alpha = dr::select(dr::neq(bsdf, nullptr), 1.f, 0.f);
        Vector2f adjusted_position = sensor_ds.uv + block->offset();

        /* Splat RGB value onto the image buffer. The particle tracer
           does not use the weight channel at all */
        block->put(adjusted_position, si.wavelengths, result, alpha,
                   /* weight = */ 0.f, active);

        return result;
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("VolumetricParticleTracerIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }


    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::select(dr::isfinite(w), w, 0.f);
    };

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VolumetricParticleTracerIntegrator, MyIntegrator);
MI_EXPORT_PLUGIN(VolumetricParticleTracerIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
