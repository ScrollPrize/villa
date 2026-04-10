-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA512

Format: 3.0 (quilt)
Source: ffmpeg
Binary: ffmpeg, ffmpeg-doc, libavcodec62, libavcodec-extra62, libavcodec-extra, libavcodec-dev, libavdevice62, libavdevice-dev, libavfilter11, libavfilter-extra11, libavfilter-extra, libavfilter-dev, libavformat62, libavformat-extra62, libavformat-extra, libavformat-dev, libavutil60, libavutil-dev, libswresample6, libswresample-dev, libswscale9, libswscale-dev
Architecture: any all
Version: 7:8.0.1-3ubuntu2
Maintainer: Debian Multimedia Maintainers <debian-multimedia@lists.debian.org>
Uploaders:  Reinhard Tartler <siretart@tauware.de>, James Cowgill <jcowgill@debian.org>, Sebastian Ramacher <sramacher@debian.org>,
Homepage: https://ffmpeg.org/
Standards-Version: 4.7.3
Vcs-Browser: https://salsa.debian.org/multimedia-team/ffmpeg
Vcs-Git: https://salsa.debian.org/multimedia-team/ffmpeg.git
Testsuite: autopkgtest
Testsuite-Triggers: build-essential, pkgconf
Build-Depends: clang [amd64 arm64 i386 ppc64el], debhelper-compat (= 13), glslang-dev, flite1-dev, frei0r-plugins-dev <!pkg.ffmpeg.stage1>, ladspa-sdk <!pkg.ffmpeg.stage1>, libaom-dev, libaribb24-dev <!pkg.ffmpeg.noextra>, libass-dev, libbluray-dev <!pkg.ffmpeg.stage1>, libbs2b-dev, libbz2-dev, libcdio-paranoia-dev, libchromaprint-dev <!pkg.ffmpeg.stage1>, libcodec2-dev, libdav1d-dev, libdvdnav-dev <!pkg.ffmpeg.stage1>, libdvdread-dev <!pkg.ffmpeg.stage1>, libdc1394-dev [linux-any], libdrm-dev [linux-any], libffmpeg-nvenc-dev [amd64 arm64 i386], libfontconfig-dev, libfreetype-dev, libfribidi-dev, libgl-dev, libgme-dev, libgnutls28-dev, libgsm1-dev, libharfbuzz-dev, libiec61883-dev [linux-any], libavc1394-dev [linux-any], libjack-jackd2-dev <!pkg.ffmpeg.stage1>, libjxl-dev <!pkg.ffmpeg.stage1>, liblilv-dev <!pkg.ffmpeg.stage1>, liblzma-dev, libmp3lame-dev, libmysofa-dev, libopenal-dev, libopencore-amrnb-dev <!pkg.ffmpeg.noextra>, libopencore-amrwb-dev <!pkg.ffmpeg.noextra>, libopenjp2-7-dev (>= 2.1), libopenmpt-dev, libopus-dev, libplacebo-dev (>= 5.229) [linux-any] <!pkg.ffmpeg.stage1>, libpulse-dev <!pkg.ffmpeg.stage1>, librabbitmq-dev <!pkg.ffmpeg.stage1>, librav1e-dev [!alpha !hppa !i386 !hurd-i386 !m68k !sh4 !sparc64 !x32] <!pkg.ffmpeg.stage1>, librist-dev <!pkg.ffmpeg.stage1>, librubberband-dev, librsvg2-dev [!alpha !hppa !hurd-i386 !m68k !sh4 !x32] <!pkg.ffmpeg.stage1>, libsctp-dev [linux-any] <!pkg.ffmpeg.stage1>, libsdl2-dev <!pkg.ffmpeg.stage1>, libshine-dev (>= 3.0.0), libsmbclient-dev (>= 4.13) [!hurd-i386] <!pkg.ffmpeg.noextra>, libsnappy-dev, libsoxr-dev, libspeex-dev, libsrt-gnutls-dev <!pkg.ffmpeg.stage1>, libssh-dev <!pkg.ffmpeg.stage1>, libsvtav1enc-dev <!pkg.ffmpeg.stage1>, libtesseract-dev <!pkg.ffmpeg.noextra>, libtheora-dev, libtwolame-dev, libva-dev (>= 1.3) [!hurd-any], libvdpau-dev, libvidstab-dev, libvo-amrwbenc-dev <!pkg.ffmpeg.noextra>, libvorbis-dev, libvpl-dev [amd64], libvpx-dev, libvulkan-dev [linux-any], libwebp-dev, libx264-dev <!pkg.ffmpeg.stage1>, libx265-dev (>= 1.8), libxcb-shape0-dev, libxcb-shm0-dev, libxcb-xfixes0-dev, libxml2-dev, libxv-dev, libxvidcore-dev, libzimg-dev, libzmq3-dev <!pkg.ffmpeg.stage1>, libzvbi-dev <!pkg.ffmpeg.stage1>, ocl-icd-opencl-dev | opencl-dev, pkgconf, texinfo, nasm, zlib1g-dev
Build-Depends-Indep: cleancss, doxygen, node-less, tree
Package-List:
 ffmpeg deb video optional arch=any
 ffmpeg-doc deb doc optional arch=all
 libavcodec-dev deb libdevel optional arch=any
 libavcodec-extra deb metapackages optional arch=any profile=!pkg.ffmpeg.noextra
 libavcodec-extra62 deb libs optional arch=any profile=!pkg.ffmpeg.noextra
 libavcodec62 deb libs optional arch=any
 libavdevice-dev deb libdevel optional arch=any
 libavdevice62 deb libs optional arch=any
 libavfilter-dev deb libdevel optional arch=any
 libavfilter-extra deb metapackages optional arch=any profile=!pkg.ffmpeg.noextra
 libavfilter-extra11 deb libs optional arch=any profile=!pkg.ffmpeg.noextra
 libavfilter11 deb libs optional arch=any
 libavformat-dev deb libdevel optional arch=any
 libavformat-extra deb metapackages optional arch=any profile=!pkg.ffmpeg.noextra
 libavformat-extra62 deb libs optional arch=any profile=!pkg.ffmpeg.noextra
 libavformat62 deb libs optional arch=any
 libavutil-dev deb libdevel optional arch=any
 libavutil60 deb libs optional arch=any
 libswresample-dev deb libdevel optional arch=any
 libswresample6 deb libs optional arch=any
 libswscale-dev deb libdevel optional arch=any
 libswscale9 deb libs optional arch=any
Checksums-Sha1:
 963990565b6599df15e23877e853ec4b0764b8ec 11388848 ffmpeg_8.0.1.orig.tar.xz
 25660c88e0313aaa9acec892d1c27022e6e3f8d3 520 ffmpeg_8.0.1.orig.tar.xz.asc
 c726f39f2d00f267281cc0facff44848d1932bf5 58084 ffmpeg_8.0.1-3ubuntu2.debian.tar.xz
Checksums-Sha256:
 05ee0b03119b45c0bdb4df654b96802e909e0a752f72e4fe3794f487229e5a41 11388848 ffmpeg_8.0.1.orig.tar.xz
 54ab4020bb4eb280444da74d7aabc282acf6e8500cf6feaefd9821373fa9e72c 520 ffmpeg_8.0.1.orig.tar.xz.asc
 a32bd75677bca735105b04a0ebd190a3481e0d8c2fcacd7d9f17524e240ee879 58084 ffmpeg_8.0.1-3ubuntu2.debian.tar.xz
Files:
 b4e86cef7b3333034977ae9df391896d 11388848 ffmpeg_8.0.1.orig.tar.xz
 c2df10a1a514a874776171960188020d 520 ffmpeg_8.0.1.orig.tar.xz.asc
 e6f0c16f441d35724520617d98ffac71 58084 ffmpeg_8.0.1-3ubuntu2.debian.tar.xz

-----BEGIN PGP SIGNATURE-----

iQIzBAEBCgAdFiEEkpeKbhleSSGCX3/w808JdE6fXdkFAmmU12oACgkQ808JdE6f
XdkcsQ/7BHhEQ6VLLjWaZK3yTJeJVOqWjIWVEoQK2Z334c0WpVhQTNvGCPnhGBWp
LBr/K+tBJtCzUKLlyPN3qTR7DRy80ZuDbuXeo2wJzbRHt4dJjgwn5nkfp1rsNHXb
Je5nNbc6AG7rFTzZKkmzYNH/88Lt/D1GxcWQBx+/f00Z0WrZMxVFunRmrEWUmqJt
mW6Q9iGJe3ULZYAIykSOOyJhS0aNBmA4xwok5r/ScJo+l4DB+373qHWh+GGFMt8G
8j/5uMPtWINQa/l9JPMP5mLEpEEvFXTdUZJDdikDjG5VYt99npKjhYP21++31BzK
ojUXkbfkZpoTEgdzABTtO43R4cTTHpe6/oBq6axebOZwXVO5xTihM/XuW69NcX6M
cvU5Uxmi0OboI7fJR23+qJ2nsHCgfFwUljHRvhGrKGK5LuahQFkjibStLCkc+rWh
EX6siuerfQKI5QB0+10IlaNRW/kHfG18YTm070YFlJQJKot7bwaagY+NYZ4Xcxz1
jJ05qH6HHP/1zVTvkB5t5yQesg415Z3YYFYEjLKSY861a6Cy2eHL0Gv5v527xGYb
4JtN4nNL1gITHyotwbxPlQ85mqfLsB0SqhnG7O/SCS/oixRpF0Vhyr8nBuElFgfa
Em7fBeppA3vrgJIRgd7pCTt7mKp8wq5Cu9ent17PS4RrqzBM9no=
=7Ze4
-----END PGP SIGNATURE-----
