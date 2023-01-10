# microtypescriptograd

This is [micrograd](https://github.com/karpathy/micrograd) in typescript, which actually has less operations than micrograd and a different backprop algorithm which doesnt use recursion since that soon fails if you use a bigger ammount of params due to reaching Max Call Stack. This was an educational project in hopes to learn more about basic multi layer perceptrons and neural nets in general, hence the sparse ammount of tests with hardcoded results.
