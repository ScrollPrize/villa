#pragma once
// Constructor parsing only; transform implementations live in their named headers.
static void volume_reference(volume_detail::ReferenceVolume& t, Arguments& a) {
    auto object=a.take("reference",nb::none());require(!object.is_none(),"reference volume is required");
    Input input;require(nb::try_cast(object,input,false),"reference must be a contiguous CPU NumPy array");
    Shape shape;for(std::size_t i=0;i<input.ndim();++i)shape.push_back(input.shape(i));
    auto dtype=input.dtype();require(dtype.code==std::uint8_t(nb::dlpack::dtype_code::Float) && (dtype.bits==32 || dtype.bits==64),"reference must be float32 or float64");
    t=volume_detail::reference_volume(Array(shape,'f',dtype.bits/8,input.data()));
}
static int volume_mode(Arguments& a,const char* name,const std::vector<std::string>& choices,const char* fallback) {
    auto mode=a.get(name,std::string(fallback));auto it=std::find(choices.begin(),choices.end(),mode);require(it!=choices.end(),std::string("invalid ")+name+": "+mode);return int(it-choices.begin());
}
static void erase_options(volume_detail::EraseValues& fill,Arguments& a) {
    fill.image=a.get("fill",fill.image);auto label=a.get("fill_mask",nb::object(fill.labels?nb::cast(fill.label):nb::none()));
    fill.labels=!label.is_none();if(fill.labels)fill.label=nb::cast<double>(label);
}
static void bind_volumetric(nb::module_& m) {
    bind<BiasField>(m,"BiasField",[](auto& t,auto& a){t.std=parameter(a,"std",t.std);t.scale=parameter(a,"scale",t.scale);});
    bind<PlasmaBrightnessContrast>(m,"PlasmaBrightnessContrast",[](auto& t,auto& a){t.brightness=parameter(a,"brightness",t.brightness);t.contrast=parameter(a,"contrast",t.contrast);t.scale=parameter(a,"scale",t.scale);t.persistence=parameter(a,"persistence",t.persistence);t.octaves=a.get("octaves",t.octaves);});
    bind<PlasmaShadow>(m,"PlasmaShadow",[](auto& t,auto& a){t.strength=parameter(a,"strength",t.strength);t.scale=parameter(a,"scale",t.scale);t.persistence=parameter(a,"persistence",t.persistence);t.octaves=a.get("octaves",t.octaves);});
    bind<Illumination>(m,"Illumination",[](auto& t,auto& a){t.strength=parameter(a,"strength",t.strength);t.mode=volume_mode(a,"mode",{"linear","corner","gaussian"},"linear");t.radius=parameter(a,"radius",t.radius);t.direction=a.get("direction",t.direction);t.center=a.get("center",t.center);});
    bind<Vignetting>(m,"Vignetting",[](auto& t,auto& a){t.strength=parameter(a,"strength",t.strength);t.radius=parameter(a,"radius",t.radius);t.center=a.get("center",t.center);});
    bind<RandomToneCurve>(m,"RandomToneCurve",[](auto& t,auto& a){t.scale=parameter(a,"scale",t.scale);});
    bind<Equalize>(m,"Equalize",[](auto& t,auto& a){t.bins=a.get("bins",t.bins);});
    bind<CLAHE>(m,"CLAHE",[](auto& t,auto& a){t.clip_limit=parameter(a,"clip_limit",t.clip_limit);t.tile_shape=a.get("tile_shape",t.tile_shape);t.bins=a.get("bins",t.bins);});
    bind<HistogramMatching>(m,"HistogramMatching",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",t.blend);t.bins=a.get("bins",t.bins);});
    bind<PixelDistributionAdaptation>(m,"PixelDistributionAdaptation",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",t.blend);t.mode=volume_mode(a,"mode",{"standard","minmax"},"standard");});
    bind<FDA>(m,"FDA",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",t.blend);t.fraction=parameter(a,"fraction",t.fraction);});
    bind<MultiplicativeNoise>(m,"MultiplicativeNoise",[](auto& t,auto& a){t.low=parameter(a,"low",t.low);t.high=parameter(a,"high",t.high);t.shared_noise=a.get("shared_noise",t.shared_noise);});
    bind<ShotNoise>(m,"ShotNoise",[](auto& t,auto& a){t.photons=parameter(a,"photons",t.photons);t.shared_noise=a.get("shared_noise",t.shared_noise);});
    bind<AdditiveNoise>(m,"AdditiveNoise",[](auto& t,auto& a){t.scale=parameter(a,"scale",t.scale);t.distribution=volume_mode(a,"distribution",{"uniform","gaussian","laplace","beta"},"uniform");t.mean=parameter(a,"mean",t.mean);t.alpha=parameter(a,"alpha",t.alpha);t.beta=parameter(a,"beta",t.beta);t.shared_noise=a.get("shared_noise",t.shared_noise);});
    bind<SaltAndPepper>(m,"SaltAndPepper",[](auto& t,auto& a){t.amount=parameter(a,"amount",t.amount);t.salt_fraction=parameter(a,"salt_fraction",t.salt_fraction);t.shared_noise=a.get("shared_noise",t.shared_noise);});
    bind<MotionBlur>(m,"MotionBlur",[](auto& t,auto& a){t.length=parameter(a,"length",t.length);t.direction=a.get("direction",t.direction);t.samples=a.get("samples",t.samples);});
    bind<Defocus>(m,"Defocus",[](auto& t,auto& a){t.radius=parameter(a,"radius",t.radius);});
    bind<ZoomBlur>(m,"ZoomBlur",[](auto& t,auto& a){t.strength=parameter(a,"strength",t.strength);t.center=a.get("center",t.center);t.samples=a.get("samples",t.samples);});
    bind<UnsharpMask>(m,"UnsharpMask",[](auto& t,auto& a){t.strength=parameter(a,"strength",t.strength);t.sigma=parameter(a,"sigma",t.sigma);});
    bind<GibbsRinging>(m,"GibbsRinging",[](auto& t,auto& a){t.cutoff=parameter(a,"cutoff",t.cutoff);});
    bind<RingingOvershoot>(m,"RingingOvershoot",[](auto& t,auto& a){t.cutoff=parameter(a,"cutoff",t.cutoff);t.radius=parameter(a,"radius",t.radius);});
    bind<ForegroundCrop>(m,"ForegroundCrop",[](auto& t,auto& a){t.shape=a.get("shape",t.shape);t.mask=a.get("mask",t.mask);t.channel=a.get("channel",t.channel);t.threshold=a.get("threshold",t.threshold);});
    bind<ChannelMisalignment>(m,"ChannelMisalignment",[](auto& t,auto& a){t.affine.degrees=parameter(a,"degrees",t.affine.degrees);t.affine.translate=parameter(a,"translate",t.affine.translate);t.affine.scale=parameter(a,"scale",t.affine.scale);t.affine.shared_scale=a.get("shared_scale",t.affine.shared_scale);t.affine.border=a.get("border",t.affine.border);});
    bind<GridDropout>(m,"GridDropout",[](auto& t,auto& a){t.unit_size=a.get("unit_size",t.unit_size);t.ratio=parameter(a,"ratio",t.ratio);t.axes=a.get("axes",t.axes);erase_options(t.fill,a);});
    bind<StripeMasking>(m,"StripeMasking",[](auto& t,auto& a){t.unit_size=a.get("unit_size",t.unit_size);t.ratio=parameter(a,"ratio",t.ratio);t.axes=a.get("axes",t.axes);erase_options(t.fill,a);});
    bind<ForegroundDropout>(m,"ForegroundDropout",[](auto& t,auto& a){t.mask=a.get("mask",t.mask);t.size=a.get("size",t.size);t.count=parameter(a,"count",t.count);t.mask_channel=a.get("mask_channel",t.mask_channel);t.threshold=a.get("threshold",t.threshold);erase_options(t.fill,a);});
    bind<ObjectDropout>(m,"ObjectDropout",[](auto& t,auto& a){t.mask=a.get("mask",t.mask);t.drop_probability=parameter(a,"drop_probability",t.drop_probability);t.mask_channel=a.get("mask_channel",t.mask_channel);t.threshold=a.get("threshold",t.threshold);erase_options(t.fill,a);});
}
