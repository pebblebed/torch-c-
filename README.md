## Compile-time shape checking for Torch Tensors, and a small demo application

We're not exactly negotiating with aliens for the future of Earth here, folks. But still, wrestling with
Tensor shapes at runtime can be annoying and slow, as it often involves getting your whole goofy stack,
dataloaders and all, into memory before you can see if you fixed it. Now you can wait for your whole goofy
compilation step to fail instead! And that might be faster for you! Always remember to reshape responsibly.
