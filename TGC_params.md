### Parameters
* $N=3$
* $S=1$
* $L=2$
* $d=6240$
* $\beta*d = \frac{d}{8} = 780$

### $TGC_{\alpha \beta}$
* $\alpha = 3$
* $\beta = \frac{1}{8}$
* $D_{n} = 3660$
* $D_{c} = 2580$
* $r_{n}(L) = \frac{d}{26} = 240$
* $r_{n}(2) = \frac{d}{26} = 240$
* $r_{c}(2) = \frac{d}{13} = 480$
* $r_{n}(1) = 500$
* $r_{c}(1) = 1000$
* Expected time for one iteration = 2280
* Coefficients
  * (1, 1): [1/2, 1] ((1/2)[860]+ (1)[140])
  * (1, 2): [1, -1] ((1)[860]+ (-1)[140])
  * (1, 3): [1/2, 1] ((1/2)[860]+ (1)[140])
  * $T_{(1, 1)}$
    * (2, 1): [1/2, 1/2, 1, 1] (4 eq partitions of size 120)
    * (2, 2): [1, 1, -1, -1]
    * (2, 3): [1/2, 1/2, 1, 1]
  * $T_{(1, 2)}$
    * (2, 4): [-1/2, -1/2, -1, -1]
    * (2, 5): [-1, -1, 1, 1]
    * (2, 6): [-1/2, -1/2, -1, -1]
  * $T_{(1, 3)}$
    * (2, 7): [1/2, 1/2, 1, 1]
    * (2, 8): [1, 1, -1, -1]
    * (2, 9): [1/2, 1/2, 1, 1]

### $TGC_{\alpha}$
* $\alpha = 3$
* $\beta = 0$
* $r_{n}(L) = \frac{2d}{39} = 320$
* $r_{n}(2) = \frac{2d}{39} = 320$
* $r_{c}(2) = \frac{4d}{39} = 640$
* $r_{n}(1) = 320$
* $r_{c}(1) = 640$
* Expected time for one iteration = 2520
* Coefficients
  * (1, 1): [1/2, 1/2]
  * (1, 2): [1, 1]
  * (1, 3): [1/2, 1/2]
  * $T_{(1, 1)}$
    * (2, 1): [1/4, 1/2, 1, 1]
    * (2, 2): [1, 1, -1, -1]
    * (2, 3): [1/4, 1/2, 1, 1]
  * $T_{(1, 2)}$
    * (2, 4): [1/2, -1/2, -1, -1]
    * (2, 5): [-1, -1, 1, 1]
    * (2, 6): [1/2, -1/2, -1, -1]
  * $T_{(1, 3)}$
    * (2, 7): [1/4, 1/2, 1, 1]
    * (2, 8): [1, 1, -1, -1]
    * (2, 9): [1/4, 1/2, 1, 1]

### $TGC_{\beta}$
* $\alpha = \inf$
* $\beta = \frac{d}{8}$
* $r_{c}(L) = \frac{13d}{60} = 1352$
* $r_{c}(2) = \frac{13d}{60} = 1352$
* $r_{c}(1) = 2132$
* Expected time for one iteration = 2912
* Coefficients
  * (1, 1): [1/2, 1] ((1/2)[2080]+ (1)[52])
  * (1, 2): [1, -1] ((1)[2080]+ (-1)[52])
  * (1, 3): [1/2, 1] ((1/2)[2080]+ (1)[52])
  * $T_{(1, 1)}$
    * (2, 1): [1/2, 1/2, 1, 1] (4 eq partitions of size 338)
    * (2, 2): [1, 1, -1, -1]
    * (2, 3): [1/2, 1/2, 1, 1]
  * $T_{(1, 2)}$
    * (2, 4): [-1/2, -1/2, -1, -1]
    * (2, 5): [-1, -1, 1, 1]
    * (2, 6): [-1/2, -1/2, -1, -1]
  * $T_{(1, 3)}$
    * (2, 7): [1/2, 1/2, 1, 1]
    * (2, 8): [1, 1, -1, -1]
    * (2, 9): [1/2, 1/2, 1, 1]


### $TGC$
* $\alpha = \inf$
* $\beta = 0$
* $r_{c}(L) = \frac{4d}{15} = 1664$
* $r_{c}(2) = \frac{4d}{15} = 1664$
* $r_{c}(1) = 1664$
* Expected time for one iteration = 3224