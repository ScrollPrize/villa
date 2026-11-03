#pragma once
// Constructor parsing only; transform implementations live in their named headers.
static void volume_reference(volume_detail::ReferenceVolume& t, Arguments& a) {
    auto object=a.take("reference",nb::none());require(!object.is_none(),"reference volume is required");
    Input input;require(nb::try_cast(object,input,false),"reference must be a contiguous CPU NumPy array");
    require((input.ndim()==3 || input.ndim()==4),"reference must have shape (C,Y,X) or (C,Z,Y,X)");
    for(std::size_t i=0;i<input.ndim();++i)t.shape.push_back(input.shape(i));
    require(product(t.shape)>0,"reference must be nonempty");
    auto dtype=input.dtype();t.wide=dtype.bits==64;require(dtype.code==std::uint8_t(nb::dlpack::dtype_code::Float) && (dtype.bits==32 || dtype.bits==64),"reference must be float32 or float64");
    Array values(t.shape,'f',dtype.bits/8,input.data());
    values.typed([&](const auto* data){for(std::size_t i=0;i<values.size();++i)require(std::isfinite(data[i]),"reference must be finite");});
    t.values=values; // Own a copy in the supplied dtype.
}
static int volume_mode(Arguments& a,const char* name,const std::vector<std::string>& choices,const char* fallback) {
    auto mode=a.get(name,std::string(fallback));auto it=std::find(choices.begin(),choices.end(),mode);require(it!=choices.end(),std::string("invalid ")+name+": "+mode);return int(it-choices.begin());
}
static void erase_options(volume_detail::EraseValues& fill,Arguments& a) {
    fill.image=a.get("fill",0.);auto label=a.get("fill_mask",nb::object(fill.labels?nb::cast(fill.label):nb::none()));
    fill.labels=!label.is_none();if(fill.labels)fill.label=nb::cast<double>(label);
    require(std::isfinite(fill.image) && std::isfinite(fill.label),"fill values must be finite");
}
static void bind_volumetric(nb::module_& m) {
    bind<BiasField>(m,"BiasField",[](auto& t,auto& a){t.std=parameter(a,"std",Uniform{0,.3});t.scale=parameter(a,"scale",.025);});
    bind<PlasmaBrightnessContrast>(m,"PlasmaBrightnessContrast",[](auto& t,auto& a){t.brightness=parameter(a,"brightness",Uniform{-.2,.2});t.contrast=parameter(a,"contrast",Uniform{0,.3});t.scale=parameter(a,"scale",.025);t.persistence=parameter(a,"persistence",.5);t.octaves=a.get("octaves",5);});
    bind<PlasmaShadow>(m,"PlasmaShadow",[](auto& t,auto& a){t.strength=parameter(a,"strength",Uniform{.1,.5});t.scale=parameter(a,"scale",.025);t.persistence=parameter(a,"persistence",.5);t.octaves=a.get("octaves",5);});
    bind<Illumination>(m,"Illumination",[](auto& t,auto& a){t.strength=parameter(a,"strength",Uniform{-.3,.3});t.mode=volume_mode(a,"mode",{"linear","corner","gaussian"},"linear");t.radius=parameter(a,"radius",.4);t.direction=a.get("direction",t.direction);t.center=a.get("center",t.center);});
    bind<Vignetting>(m,"Vignetting",[](auto& t,auto& a){t.strength=parameter(a,"strength",Uniform{.1,.5});t.radius=parameter(a,"radius",.8);t.center=a.get("center",t.center);});
    bind<RandomToneCurve>(m,"RandomToneCurve",[](auto& t,auto& a){t.scale=parameter(a,"scale",.1);});
    bind<Equalize>(m,"Equalize",[](auto& t,auto& a){t.bins=a.get("bins",256);});
    bind<CLAHE>(m,"CLAHE",[](auto& t,auto& a){t.clip_limit=parameter(a,"clip_limit",2.);t.tile_shape=a.get("tile_shape",t.tile_shape);t.bins=a.get("bins",256);});
    bind<HistogramMatching>(m,"HistogramMatching",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",Uniform{.5,1});t.bins=a.get("bins",256);});
    bind<PixelDistributionAdaptation>(m,"PixelDistributionAdaptation",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",Uniform{.5,1});t.mode=volume_mode(a,"mode",{"standard","minmax"},"standard");});
    bind<FDA>(m,"FDA",[](auto& t,auto& a){volume_reference(t.reference,a);t.blend=parameter(a,"blend",Uniform{.5,1});t.fraction=parameter(a,"fraction",.1);});
    bind<MultiplicativeNoise>(m,"MultiplicativeNoise",[](auto& t,auto& a){t.low=parameter(a,"low",.9);t.high=parameter(a,"high",1.1);t.shared_noise=a.get("shared_noise",false);});
    bind<ShotNoise>(m,"ShotNoise",[](auto& t,auto& a){t.photons=parameter(a,"photons",Uniform{50,200});t.shared_noise=a.get("shared_noise",false);});
    bind<AdditiveNoise>(m,"AdditiveNoise",[](auto& t,auto& a){t.scale=parameter(a,"scale",.05);t.distribution=volume_mode(a,"distribution",{"uniform","gaussian","laplace","beta"},"uniform");t.mean=parameter(a,"mean",0.);t.alpha=parameter(a,"alpha",2.);t.beta=parameter(a,"beta",2.);t.shared_noise=a.get("shared_noise",false);});
    bind<SaltAndPepper>(m,"SaltAndPepper",[](auto& t,auto& a){t.amount=parameter(a,"amount",.01);t.salt_fraction=parameter(a,"salt_fraction",.5);t.shared_noise=a.get("shared_noise",true);});
    bind<MotionBlur>(m,"MotionBlur",[](auto& t,auto& a){t.length=parameter(a,"length",Uniform{1,5});t.direction=a.get("direction",t.direction);t.samples=a.get("samples",9);});
    bind<Defocus>(m,"Defocus",[](auto& t,auto& a){t.radius=parameter(a,"radius",Uniform{1,3});});
    bind<ZoomBlur>(m,"ZoomBlur",[](auto& t,auto& a){t.strength=parameter(a,"strength",Uniform{0,.2});t.center=a.get("center",t.center);t.samples=a.get("samples",9);});
    bind<UnsharpMask>(m,"UnsharpMask",[](auto& t,auto& a){t.strength=parameter(a,"strength",Uniform{.1,.5});t.sigma=parameter(a,"sigma",Uniform{.5,1.5});});
    bind<GibbsRinging>(m,"GibbsRinging",[](auto& t,auto& a){t.cutoff=parameter(a,"cutoff",Uniform{.5,.9});});
    bind<RingingOvershoot>(m,"RingingOvershoot",[](auto& t,auto& a){t.cutoff=parameter(a,"cutoff",Uniform{.5,.9});t.radius=parameter(a,"radius",4.);});
    bind<ForegroundCrop>(m,"ForegroundCrop",[](auto& t,auto& a){t.shape=a.get("shape",t.shape);t.mask=a.get("mask",t.mask);t.channel=a.get("channel",t.channel);t.threshold=a.get("threshold",0.);require((t.shape.size()==2 || t.shape.size()==3) && product(t.shape)>0,"shape needs two or three positive sizes");require(std::isfinite(t.threshold),"threshold must be finite");});
    bind<ChannelMisalignment>(m,"ChannelMisalignment",[](auto& t,auto& a){t.affine.degrees=parameter(a,"degrees",Uniform{-5,5});t.affine.translate=parameter(a,"translate",Uniform{-2,2});t.affine.scale=parameter(a,"scale",Uniform{.95,1.05});t.affine.shared_scale=a.get("shared_scale",false);t.affine.border=a.get("border",false);});
    bind<GridDropout>(m,"GridDropout",[](auto& t,auto& a){t.unit_size=a.get("unit_size",t.unit_size);t.ratio=parameter(a,"ratio",t.ratio);t.axes=a.get("axes",t.axes);erase_options(t.fill,a);});
    bind<StripeMasking>(m,"StripeMasking",[](auto& t,auto& a){t.unit_size=a.get("unit_size",t.unit_size);t.ratio=parameter(a,"ratio",t.ratio);t.axes=a.get("axes",t.axes);erase_options(t.fill,a);});
    bind<ForegroundDropout>(m,"ForegroundDropout",[](auto& t,auto& a){t.mask=a.get("mask",t.mask);t.size=a.get("size",t.size);t.count=parameter(a,"count",t.count);t.mask_channel=a.get("mask_channel",t.mask_channel);t.threshold=a.get("threshold",t.threshold);require(std::isfinite(t.threshold),"threshold must be finite");erase_options(t.fill,a);});
    bind<ObjectDropout>(m,"ObjectDropout",[](auto& t,auto& a){t.mask=a.get("mask",t.mask);t.drop_probability=parameter(a,"drop_probability",t.drop_probability);t.mask_channel=a.get("mask_channel",t.mask_channel);t.threshold=a.get("threshold",t.threshold);require(std::isfinite(t.threshold),"threshold must be finite");erase_options(t.fill,a);});
}
