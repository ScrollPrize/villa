# vesuvius-c

From [Vesuvius Challenge](https://scrollprize.org), a single-header C library for accessing CT scans of ancient scrolls.

`vesuvius-c` allows direct access to scroll data **without** managing download scripts or storing terabytes of CT scans locally:

```c
#include "vesuvius-c.h"

int main() {
    //pick a region in the scoll to visualize
    int vol_start[3] = {3072,3072,3072};
    int chunk_dims[3] = {128,512,512};
    
    //initialize the volume
    volume* scroll_vol = vs_vol_new(
        "./54keV_7.91um_Scroll1A.zarr/0/",
        "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/");
    
    // get the scroll data by reading it from the cache and downloading it if necessary
    chunk* scroll_chunk = vs_vol_get_chunk(scroll_vol, vol_start,chunk_dims);

    // Fetch a slice  from the volume
    slice* myslice = vs_slice_extract(scroll_chunk, 0);

    // Write slice image to file
    vs_bmp_write("xy_slice.bmp",myslice);
    
    return 0;
}
```

Resulting image:

![slice](https://github.com/user-attachments/assets/f9fe5667-41e6-49f3-9e15-ca1d366ce293)

Vesuvius-c can be used to work with zarr volumes, modify them, and write out the result in various formats. This video demonstrates [christmas tree highlighting](example2.c#L94) using a combination of scroll volume and surface segmentation provided by [@bruniss](https://github.com/bruniss):

https://github.com/user-attachments/assets/e4b90221-744a-46d6-a7c2-cc3f1685fd54


The library fetches scroll data from the Vesuvius Challenge [data server](https://dl.ash2txt.org) in the background. Only the necessary volume chunks are requested, and on-disk caching is used to store downloads between program invocations and avoid re-downloading identical files.

For a similar library in Python, see [vesuvius](https://github.com/ScrollPrize/vesuvius).

> ⚠️ `vesuvius-c`  is in beta and the interface may change. Please feel free to reach out with development ideas.

## Usage

Please accept the [data agreement](https://forms.gle/HV1J6dJbmCB2z5QL8) before use.

See [example.c](example.c) for example library usage.

## Building

### Dependencies:

* [libcurl](https://curl.se/libcurl/)
* [json-c](https://json-c.github.io/json-c/)
* [c-blosc2](https://github.com/Blosc/c-blosc2)
* [ffmpeg](https://www.ffmpeg.org/) (optional)

`libcurl` is used for fetching volume chunks and is likely already available on your system. `c-blosc2` is used to decompress the Zarr chunks read from the server and may require installation. `json-c` is used to read the zarr metadata. `ffmpeg` is used to generate video from chunk data.

### Installing Dependencies

On Ubuntu, dependencies can be installed with

```sh
#install tools
sudo apt install gcc build-essential cmake ffmpeg

#install development libraries
sudo apt install zlib1g zlib1g-dev liblz4-dev libblosc2-dev  libcurl4-openssl-dev

#install dependencies from source
git clone https://github.com/json-c/json-c.git
cd json-c
mkdir build
cd build
cmake ..
make
sudo make install
```

### Build and run:

Link the dependencies and build your program:

```sh
gcc -o example example.c -lcurl -lblosc2 -ljson-c
./example
```

It may be necessary to point to the `c-blosc2` installation. For example, on Apple Silicon after `brew install c-blosc2`:

```sh
gcc -o example example.c -I/opt/homebrew/Cellar/c-blosc2/2.15.1/include -L/opt/homebrew/Cellar/c-blosc2/2.15.1/lib -lcurl -lblosc2 -ljson-c
./example
```

It may also be necessary to link with the system math library:

```sh
gcc -o example example.c -lcurl -lblosc2 -ljson-c -lm
./example
```

Vesuvius-c also has a CMakeLists.txt that will automatically discover and link the necessary libraries:

```sh
cd vesuvius-c
mkdir build
cd build
cmake ..
make
./vesuvius_example
```
