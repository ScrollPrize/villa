# LOG

pretty interesting result with/without monotoniciy

~~~
commit 09496cca7d46ab2dd2725561aaa32ae117a0293b (HEAD -> main)
Author: Hendrik Schilling <hendrik.schilling@posteo.de>
Date:   Wed Nov 26 12:44:39 2025 +0100

    monotonicity loss
~~~

- with: not so great result, but no "invalid" geometry
- without: quite good fit, clear "jumps" where it doesn't work - also adds "invalid" dead ends by repeating the same cos period!

[hendrik@staticsheep exps_2d]$ python fit_cosine_grid.py --image inf_sample1.tif --output-prefix res/res --downscale 4 --lambda-smooth 10 --device cuda --snapshot 1000 --lr 0.01 --steps 10000 --lambda-mono 100
31a05dbb7c076ff8db3c559f3ed883d2d28feff3
-> pretty good!
