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
class VolumetricParticleTracerIntegrator2 final : public MyIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MyIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_max_depth)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr, Medium, MediumPtr, PhaseFunctionContext)

    VolumetricParticleTracerIntegrator2(const Properties &props) : Base(props) { }

    MI_INLINE
    bool isZero(Spectrum s) const {
        for (int i=0; i<3; i++) {
            if (s[i] != 0.0f)
                return false;
        }
        return true;
    }

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

    // inline auto maximum(Spectrum s) const {
    //     auto max = 0;
    //     for (int i=0; i<3; i++) {
    //         if (s[i] > max)
    //             max = s[i];
    //     }
    //     return max;
    // }

    // inline Spectrum normalise(Spectrum s) const {
    //     Float max = 0.0f;
    //     Spectrum result;
    //     for (int i=0; i<3; i++) {
    //         if (s[i] > max) max = s[i];
    //     }

    //     if(max != 0.0f){
    //     for (int i=0; i<3; i++) {
    //         result[i] = s[i] / max;
    //     }
    //     }   
    //     return result;
    // }

//     Spectrum sampleAttenuatedSensorDirect(DirectionSample3f &dRec,
//         const Medium *medium, int &interactions, const Point2f &sample, Sampler *sampler, const Sensor *sensor, Mask active) const {
    
//     Spectrum value = sensor->sample_direction(dRec, sampler->next_2d(), active);

//     // if (dRec.pdf != 0) {
//     //     value *= evalTransmittance(dRec.ref, false, dRec.p, m_sensor->isOnSurface(),
//     //         dRec.time, medium, interactions, sampler);
//     //     dRec.object = m_sensor.get();
//     //     return value;
//     // } else {
//         return Spectrum(0.0f);
//     //}
// }

    void sample(const Scene *scene, const Sensor *sensor, Sampler *sampler,
                ImageBlock *block, ScalarFloat sample_scale, const Medium *initial_medium) const override { //,const Medium *initial_medium) const override {
        // Account for emitters directly visible from the sensor
        if (m_max_depth != 0 && !m_hide_emitters)
            sample_visible_emitters(scene, sensor, sampler, block, sample_scale);

        // Primary & further bounces illumination
        auto [ray, throughput] = prepare_ray(scene, sensor, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = dr::neq(throughput_max, 0.f);

        trace_light_ray(ray, scene, sensor, sampler, throughput, block,
                        sample_scale, initial_medium, active);
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
        //connect_sensor(scene, si, sensor_ds, nullptr, weight, block, sample_scale, active);
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

    
    void handleMediumInteraction2(Int32 depth, int nullInteractions, bool caustic,
        const MediumInteraction3f mei, MediumPtr medium, const Vector3f &wi,
        const Spectrum weight, ImageBlock *block, Mask active, Sampler *sampler, const Sensor *sensor, ScalarFloat sample_scale) const{

        Mask depthReached = true;
        depthReached &= depth >= m_max_depth;
        depthReached &= m_max_depth > 0;
        if (dr::any_or<true>(depthReached))
            return;
        
        //int maxInteractions = m_max_depth - depth - 1;

        Spectrum value = weight * Spectrum(0.1f);

        }

    void handleMediumInteraction(Int32 depth, int nullInteractions, bool caustic,
        const MediumInteraction3f mei, MediumPtr medium, const Vector3f &wi,
        const Spectrum weight, ImageBlock *block, Mask active, Sampler *sampler, const Sensor *sensor, ScalarFloat sample_scale, SurfaceInteraction3f si) const{

    // Mask thingy = true;
    // thingy &= depth >= m_max_depth;
    // thingy &= m_max_depth > 0;
    // if (dr::any_or<true>(thingy))
    //     return;
    
    Float aovs[5];

    auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, sampler->next_2d(), active);
    //auto [sensor_ds, sensor_weight] = sensor->sample_direction(mei, sampler->next_2d(), true);
    

    //Spectrum value = weight * sample_scale;
    Spectrum value = 0.f;
    Spectrum surface_weight = 1.f;
    value = weight;
    // value[0] = 0.444444f;
    // value[1] = 0.444444f;
    // value[2] = 0.444444f;
    // auto max = maximum(value);
    // if (max != 0.0f):
    //     value[0] /= max;
    //     value[1] /= max;
    //     value[3] /= max;

    //value[0] /= 10000;
    //value[1] /= 10000;
    //value[2] /= 10000;

    //dr::masked(value, (value > 1)) = dr::zeros<Spectrum>();
   //std::cout << value;


    if (isZero(value))
        return;
    
    
    /* Evaluate the phase function */
    auto phase = mei.medium->phase_function();
    PhaseFunctionContext phase_ctx(sampler);
    auto [wo, phase_weight, phase_pdf] = phase->sample(phase_ctx, mei, sampler->next_1d(true),
                    sampler->next_2d(true),
                    //sensor_ds.d,
                    true);

    if (isZero(value))
        return;


    
    UnpolarizedSpectrum spec_u = unpolarized_spectrum(value);
    wavelength_t<Spectrum> wavs;
    Color3f rgb;
    if constexpr (is_spectral_v<Spectrum>){
        //rgb = spectrum_to_srgb(spec_u, ray.wavelengths, active);
        rgb = spectrum_to_srgb(spec_u, wavs, active);}
    else if constexpr (is_monochromatic_v<Spectrum>){
        rgb = spec_u.x();}
    else{ // what does else imply, i.e what is is_spectral and is_monochromatic
        rgb = spec_u;}

    //std::cout << rgb << "\n";
   aovs[0] = rgb.x();
    aovs[1] = rgb.y();
    aovs[2] = rgb.z();
    // aovs[0] = 0.f;
    // aovs[1] = 0.f;
    // aovs[2] = 0.f;
    aovs[3] = 1.f;
    //aovs[3] = dr::select(true, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

   //Vector2f adjusted_position = sensor_ds.uv + block->offset();
   //block->put(adjusted_position, aovs, active);

   //std::cout << aovs << "\n";
   Float alpha = 1.f;
   Vector2f adjusted_position = sensor_ds.uv + block->offset();

    //std::cout << "made it here " << value << "\n";
        /* Splat RGB value onto the image buffer. The particle tracer
           does not use the weight channel at all */
    //block->put(adjusted_position, mei.wavelengths, value, alpha,
                   ///* weight = */ 0.f, active);
    //std::cout << value << "\n";
    block->put(adjusted_position, aovs, false);
} 
 

    std::pair<Spectrum, Float>
    trace_light_ray(Ray3f ray, const Scene *scene, const Sensor *sensor,
                    Sampler *sampler, Spectrum throughput, ImageBlock *block,
                    ScalarFloat sample_scale, MediumPtr medium, Mask active = true) const {
        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Int32 depth = 1;

        /* ---------------------- Path construction ------------------------- */
        // First intersection from the emitter to the scene
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) dr::array_size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        active &= si.is_valid();
        if (m_max_depth >= 0)
            active &= depth < m_max_depth;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Particle Tracer Integrator", active, depth, ray,
                            throughput, si, eta, sampler, mei, channel);

        int nullInteractions = 0;
        bool delta = false;
        //Spectrum throughput(1.0f);
        // Incrementally build light path using BSDF sampling.
        while (loop(active)) {
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask isvalid;

            // BSDFPtr bsdf = si.bsdf(ray);
            //     auto [sensor_ds, sensor_weight] =
            //         sensor->sample_direction(si, sampler->next_2d(), active);
            
            BSDFPtr bsdf = si.bsdf(ray);
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, sampler->next_2d(), active);
            connect_sensor(scene, si, sensor_ds, bsdf,
                           throughput * sensor_weight, block, sample_scale,
                           active, active_medium, mei, sampler);


            si = scene->ray_intersect(ray, active);
            if (dr::any_or<true>(active_medium)){
                mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                isvalid = mei.is_valid();
            }
            else {isvalid = false;}

            if (dr::any_or<true>(active_medium) && dr::any_or<true>(isvalid)) {
                
                auto [tr, free_flight_pdf] = medium->transmittance_eval_pdf(mei, si, active_medium);
                Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                //std::cout << mei.sigma_s << "\n";
                //throughput *= mei.sigma_s * tr / free_flight_pdf;
                // throughput *= mei.sigma_s * index_spectrum(mei.combined_extinction, channel) / index_spectrum(mei.sigma_t, channel);
                //throughput *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);

                // BSDFPtr bsdf = si.bsdf(ray);
                // auto [sensor_ds, sensor_weight] =
                //     sensor->sample_direction(si, sampler->next_2d(), active);
                // connect_sensor(scene, si, sensor_ds, bsdf,
                //             throughput * sensor_weight, block, sample_scale,
                //             active, active_medium, mei, sampler);
                    
                //handleMediumInteraction(depth, nullInteractions, delta, mei, medium, -ray.d, throughput, block, active, sampler, sensor, sample_scale, si);

                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mei.medium->phase_function();

                //std::cout << phase << "\n";

                //auto [phase_val, phase_pdf] = phase->eval_pdf(phase_ctx, mei, -ray.d, active);
                // change to -ray.d ?
                auto [wo, phase_weight, _] = phase->sample(phase_ctx, mei,
                    sampler->next_1d(active),
                    sampler->next_2d(active),
                    active);

                //std::cout << phase_val << "\n";
                //throughput *= phase_weight;
                delta = false;

                //std::cout << throughput << "\n";

                ray = mei.spawn_ray(wo);

            }
            //else if (si.t == std::numeric_limits<Float>::infinity()) { break;}
            else {
            
           // BSDFPtr bsdf = si.bsdf(ray);
            //     auto [sensor_ds, sensor_weight] =
            //         sensor->sample_direction(si, sampler->next_2d(), active);
            // connect_sensor(scene, si, sensor_ds, bsdf,
            //                 throughput * sensor_weight, block, sample_scale,
            //                 active, active_medium, mei, sampler);

            /* ----------------------- BSDF sampling ------------------------ */
            // Sample BSDF * cos(theta).
            // is any or true active medium the right thing
            if (dr::any_or<true>(active_medium)) {
                auto [tr, free_flight_pdf] = medium->transmittance_eval_pdf(mei, si, active_medium);
                Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                //throughput *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                //std::cout << free_flight_pdf;
                //throughput *= tr / (1 - free_flight_pdf);
            }
            
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

            if (dr::any_or<true>(si.is_medium_transition()))
                    //std::cout << "it is medium transition";
                medium = si.target_medium(wo_dot_geo_n);

            // Intersect the BSDF ray against scene geometry (next vertex).
            ray = si.spawn_ray(si.to_world(bs.wo));
            }
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
                            Mask active, Mask active_medium, MediumInteraction3f mei, Sampler *sampler) const {
        active &= (sensor_ds.pdf > 0.f) &&
                  dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
        if (dr::none_or<false>(active))
            return 0.f;

        // Check that sensor is visible from current position (shadow ray).
        Ray3f sensor_ray = si.spawn_ray_to(sensor_ds.p);
        active &= !scene->ray_test(sensor_ray, active);
        if (dr::none_or<false>(active) && dr::none_or<false>(active_medium))
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
        // Mask not_on_surface = active && dr::eq(si.shape, nullptr) && dr::eq(bsdf, nullptr);
        // if (dr::any_or<true>(not_on_surface)) {
        //     Mask invalid_side = Frame3f::cos_theta(local_d) <= 0.f;
        //     surface_weight[not_on_surface && invalid_side] = 0.f;
        // }

        if(dr::any_or<true>(active_medium)){
            // PhaseFunctionContext phase_ctx(sampler);
            //     auto phase = mei.medium->phase_function();
            // auto [wo, phase_weight, _] = phase->sample(phase_ctx, mei,
            //         sampler->next_1d(active),
            //         sampler->next_2d(active),
            //         active);
            
           result = weight/4 * surface_weight * sample_scale;//*phase_weight;
        //std::cout << result << "\n";

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
        }
        else {
        result = weight * surface_weight * sample_scale;
        //std::cout << result << "\n";

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
        }
        

        return result;
    }

    Spectrum connect_sensor2(const Scene *scene, const SurfaceInteraction3f &si,
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
        //if (dr::none_or<false>(active))
            //return 0.f;

        // Foreshortening term and BSDF value for that direction (for surface interactions).
        Spectrum result = 0.f;
        Spectrum surface_weight = 1.f;
        Vector3f local_d        = si.to_local(sensor_ray.d);
        Mask on_surface         = active && dr::neq(si.shape, nullptr);
        if (dr::any_or<true>(on_surface)) {
            /* Note that foreshortening is only missing for directly visible
               emitters associated with a shape. Otherwise it's included in the
               BSDF. Clamp negative cosines (zero value if behind the surface). */

            // surface_weight[on_surface && dr::eq(bsdf, nullptr)] *=
            //     dr::maximum(0.f, Frame3f::cos_theta(local_d));
            surface_weight[on_surface && dr::eq(bsdf, nullptr)] *= Frame3f::cos_theta(local_d);

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
        // Mask not_on_surface = active && dr::eq(si.shape, nullptr) && dr::eq(bsdf, nullptr);
        // if (dr::any_or<true>(not_on_surface)) {
        //     Mask invalid_side = Frame3f::cos_theta(local_d) <= 0.f;
        //     surface_weight[not_on_surface && invalid_side] = 0.f;
        // }

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
        return tfm::format("VolumetricParticleTracerIntegrator2[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VolumetricParticleTracerIntegrator2, MyIntegrator);
MI_EXPORT_PLUGIN(VolumetricParticleTracerIntegrator2, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)