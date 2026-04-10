-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA512

Format: 3.0 (quilt)
Source: x264
Binary: x264, libx264-165, libx264-dev
Architecture: any
Version: 2:0.165.3222+gitb35605ac-3build1
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Uploaders:  Reinhard Tartler <siretart@tauware.de>, Rico Tzschichholz <ricotz@ubuntu.com>, Sebastian Ramacher <sramacher@debian.org>
Homepage: https://www.videolan.org/developers/x264.html
Standards-Version: 4.7.2
Vcs-Browser: https://salsa.debian.org/multimedia-team/x264
Vcs-Git: https://salsa.debian.org/multimedia-team/x264.git
Testsuite: autopkgtest
Testsuite-Triggers: ffmpeg
Build-Depends: debhelper-compat (= 13), libavformat-dev (>= 6:9) <!stage1>, libffms2-dev <!stage1>, nasm (>= 2.13) [any-i386 any-amd64], pkgconf
Package-List:
 libx264-165 deb libs optional arch=any
 libx264-dev deb libdevel optional arch=any
 x264 deb video optional arch=any profile=!stage1
Checksums-Sha1:
 6a9d8cdc1e877b6ee50293746d0478268a475d13 1038624 x264_0.165.3222+gitb35605ac.orig.tar.gz
 32fbe386ee1836acde2c930804b2432c4f279494 23292 x264_0.165.3222+gitb35605ac-3build1.debian.tar.xz
Checksums-Sha256:
 4672fb415c34bf16e2ed9cd43d1ab865158f586c7f1406d507f7f44516fb5ec8 1038624 x264_0.165.3222+gitb35605ac.orig.tar.gz
 28bb2b111c288cbcbd844da4577203e87653b769190a10ab078b5b8c1c58612f 23292 x264_0.165.3222+gitb35605ac-3build1.debian.tar.xz
Files:
 f405c8d52d361f3ccf2b7389d3acc09b 1038624 x264_0.165.3222+gitb35605ac.orig.tar.gz
 69647e871d76b4093cf9ec2dba1f6240 23292 x264_0.165.3222+gitb35605ac-3build1.debian.tar.xz
Original-Maintainer: Debian Multimedia Maintainers <debian-multimedia@lists.debian.org>

-----BEGIN PGP SIGNATURE-----

iQJFBAEBCgAvFiEEDMvPrK69u5wsjzS2IqBL75FoutUFAmlOj4QRHHJpa21pbGxz
QGtkZS5vcmcACgkQIqBL75FoutUflA/9FyatjDb+Un6aNGiU8YWHg3eCj9ieyiJr
sqnCjYKXvL8BDlcrs3VSPA0Dw53cB3+XMcqnMIZ7aQj4KLECZR8PKPFQZMABSu46
vQ9mtCZACsKEiUE3cC3wlxK87+tP0Z9JEB1VWUNvMhaG6jeLWHjM+hHD/TEH77J1
h95ThMFxlEQezGl2Tn614RHCAiOWddjC0qXXpIrxLFHCnFmNPAjZK/svrAAbSePw
LkeTjmtDuFcBzl4GPVtxDPNvt0v8S9pisky3Kf218eoiUdY9hQpvKdDIrBOomnI1
9OtIqlqo4SvFar4N1jQ3BarxM2yLtZf5mD6MR2tPTIyx/VAPM/UE11/BgVkBzWvT
qCDRmI8jFdeCntvZR3WPlclEJAwB/DV2yWitqJUi5bp5pGUBsPH6ZmiCSE0caula
ssPdEiCuEKt3FXew4IIALUsLwfJC10HovCl5KP3h72c3rgU7m47j51C5crVQ0MAH
nZSgdnjnl0RLy5CiGoAekfthe6cLgFMNBXBZta5wNRF3I7Ab22TRZM2s/6Wj3svj
iHRM1tYKRgXALMt1rABNfit92f5qi+E90kS00//PmckY2+CzOkTn6FI9uwPESmmG
l58usRjcaoWf1pPLUCZzoZWHLwiic5d3syy7azHXtFnHRUiE0mnsOucnTWLdr5SU
VdjWSVPejOM=
=rPIm
-----END PGP SIGNATURE-----
