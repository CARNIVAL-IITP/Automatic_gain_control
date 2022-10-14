from .modelwrapper import ModelWrapper_, AudioModelWrapper

def get_wrapper(model : str) -> ModelWrapper_:
    ### Audio ###
    if model == "hifigan":
        from .hifigan import ModelWrapper
    elif model == "hifigan_loss":
        from .hifigan_loss import ModelWrapper
    elif model == "hifigan_causal":
        from .hifigan_causal import ModelWrapper
    
    elif model == "speedyspeech":
        from .speedyspeech.default import ModelWrapper
    elif model == "speedyspeech_dur":
        from .speedyspeech.dur import ModelWrapper
    elif model == "speedyspeech_dws":
        from .speedyspeech.dws import ModelWrapper
    elif model == "speedyspeech_norm":
        from .speedyspeech.norm import ModelWrapper
    elif model == "speedyspeech_conv":
        from .speedyspeech.conv import ModelWrapper
    elif model == "speedyspeech_conv2":
        from .speedyspeech.conv2 import ModelWrapper
    elif model == "speedyspeech_conv3":
        from .speedyspeech.conv3 import ModelWrapper
    elif model == "speedyspeech_conv3_dws":
        from .speedyspeech.conv3_dws import ModelWrapper
    
    elif model == "soundstream":
        from .soundstream.default import ModelWrapper
    elif model == "soundstream_mag":
        from .soundstream.mag import ModelWrapper
    elif model == "soundstream_spec_loss":
        from .soundstream.spec_loss import ModelWrapper
    elif model == "soundstream_nostft":
        from .soundstream.nostft import ModelWrapper
    elif model == "soundstream_magtest":
        from .soundstream.magtest import ModelWrapper
    elif model == "soundstream_norm":
        from .soundstream.norm import ModelWrapper
    elif model == "soundstream_novq":
        from .soundstream.novq import ModelWrapper
    elif model == "soundstream_mpd":
        from .soundstream.mpd import ModelWrapper
    elif model == "soundstream_wn":
        from .soundstream.wn import ModelWrapper
    elif model == "soundstream2_bn":
        # downsampling -> resblock
        # lower performance
        from .soundstream.bn2 import ModelWrapper
    elif model == "soundstream3_bn":
        # resblock -> downsampling, as usual
        # better performance w.r.t. ver.2
        from .soundstream.bn3 import ModelWrapper
    elif model == "soundstream3_ws":
        from .soundstream.ws3 import ModelWrapper
    elif model == "soundstream3_ws_nostft":
        from .soundstream.ws3_nostft import ModelWrapper
    elif model == "soundstream_1d":
        from .soundstream.ss1d import ModelWrapper
    elif model == "soundstream_1d_magspec":
        from .soundstream.ss1d_magspec import ModelWrapper
    elif model == "soundstream_1d_scale":
        from .soundstream.ss1d_scale import ModelWrapper
    elif model == "soundstream_1d_loss":
        from .soundstream.ss1d_loss import ModelWrapper
    elif model == "soundstream_1d_grad":
        from .soundstream.ss1d_grad import ModelWrapper
    elif model == "soundstream_1d_pm":
        from .soundstream.ss1d_pm import ModelWrapper
    elif model == "soundstream_1d_nmr":
        from .soundstream.ss1d_nmr import ModelWrapper
    elif model == "soundstream_xlogx_slaney":
        from .soundstream.xlogx_slaney import ModelWrapper
    elif model == "soundstream_xlogx_none":
        from .soundstream.xlogx_none import ModelWrapper
    elif model == "soundstream_ws_1d":
        from .soundstream.ws_1d import ModelWrapper
    elif model == "soundstream_ws_1d_fast":
        from .soundstream.ws_1d_fast import ModelWrapper
    elif model == "soundstream_logdelta":
        from .soundstream.logdelta import ModelWrapper
    elif model == "soundstream_dcnorm":
        from .soundstream.dcnorm import ModelWrapper
    elif model == "soundstream_wavegan":
        from .soundstream.wavegan import ModelWrapper
    
    elif model == "tinytts":
        from .tinytts.default import ModelWrapper
    elif model == "tinytts_gradnorm":
        from .tinytts.gradnorm import ModelWrapper
    elif model == "tinytts_v3":
        from .tinytts.v3 import ModelWrapper
    elif model == "tinytts_deep_v1":
        from .tinytts.deep_v1 import ModelWrapper
    elif model == "tinytts_deep_v2":
        from .tinytts.deep_v2 import ModelWrapper
    
    elif model == "postgan":
        from .postgan.default import ModelWrapper
    elif model == "postgan_mpd":
        from .postgan.mpd import ModelWrapper
    elif model == "postgan_mpd2":
        from .postgan.mpd2 import ModelWrapper
    elif model == "postgan_mrd":
        from .postgan.mrd import ModelWrapper
    elif model == "postgan_tg":
        from .postgan.tg import ModelWrapper
    elif model == "postgan_unet":
        from .postgan.unet import ModelWrapper
    elif model == "postgan_old":
        from .postgan.old import ModelWrapper
    elif model == "postgan_smallcond":
        from .postgan.smallcond import ModelWrapper
    elif model == "postgan_stft":
        from .postgan.stft import ModelWrapper
    elif model == "postgan_ss":
        from .postgan.ss import ModelWrapper
    elif model == "postgan_ss_mrd":
        from .postgan.ss_mrd import ModelWrapper
    
    elif model == "sfnet_mdct":
        from .sfnet.mdct import ModelWrapper
    elif model == "sfnet_mdct_learnable":
        from .sfnet.mdct_learnable import ModelWrapper
    elif model == "sfnet_mdct_conv":
        from .sfnet.mdct_conv import ModelWrapper
    elif model == "sfnet_stft":
        from .sfnet.stft import ModelWrapper
    
    elif model == "dccrn":
        from .dccrn.default import ModelWrapper
    elif model == "dctcrn":
        from .dctcrn.default import ModelWrapper
    elif model == "dctcrn_conv":
        from .dctcrn.conv import ModelWrapper
    elif model == "dctcrn_ar":
        from .dctcrn.ar import ModelWrapper
    elif model == "dctcrn_ar_trick":
        from .dctcrn.ar_trick import ModelWrapper
    elif model == "dctcrn_ar_trick_sampling":
        from .dctcrn.ar_trick_sampling import ModelWrapper
    elif model == "dctcrn_ar_minimal_jit":
        from .dctcrn.ar_minimal_jit import ModelWrapper
    elif model == 'dctcrn_two_way':
        from .dctcrn.two_way import ModelWrapper
    elif model == 'dctcrn_two_encoder':
        from .dctcrn.two_encoder import ModelWrapper
    elif model == 'with_asa':
        from .dctcrn.with_asa import ModelWrapper
    elif model == 'echofilter_like':
        from .dctcrn.echofilter_like import ModelWrapper
    elif model == 'with_wav2vec':
        from .dctcrn.with_wav2vec import ModelWrapper
    elif model == 'with_gated':
        from .dctcrn.with_gated import ModelWrapper

    elif model == 'echofilter':
        from .echofilter.default import ModelWrapper
    elif model == 'echofilter_two_way':
        from .echofilter.two_way import ModelWrapper
    elif model == 'echofilter_two_way_emerge':
        from .echofilter.two_way_emerge import ModelWrapper
    elif model == 'echofilter_with_s4d':
        from .echofilter.with_s4d import ModelWrapper
    elif model == 'echofilter_with_attention':
        from .echofilter.with_attention import ModelWrapper
    elif model == 'EDLMR':
        from .EDLMR.default import ModelWrapper
        
        
        
    elif model == 'complex':
        from .complex.default import ModelWrapper
    elif model == 'complex_with_simple_decoder':
        from .complex.decoder_change import ModelWrapper
        
    elif model == 'complex_light_model':
        from .complex.small import ModelWrapper
    
    elif model == 'complex_dct':
        from .complex.with_dct import ModelWrapper
        
    elif model == 'simple_merge_branch':
        from .complex.with_simple_merge import ModelWrapper
        
    elif model == 'scale':
        from .complex.scale import ModelWrapper
    
    elif model == 'masking':
        from .complex.masking import ModelWrapper
    
    elif model == 'merge':
        from .complex.merge import ModelWrapper
        
    elif model =='sadnunet':
        from .sadnunet.default import ModelWrapper
    elif model == 'sadnunet_dct':
        from .sadnunet.dct import ModelWrapper
    else:
        raise NotImplementedError(f"model '{model}' is not implemented")
    return ModelWrapper