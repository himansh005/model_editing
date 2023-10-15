<<<<<<< HEAD
# model_editing

1. Test generation on counterfact dataset. (ensure pre-edit predictions are right)
2. Write code that does the following:

- Consider <subject> <relation> <object> type edits
- At every step, perform forward pass through the 2 models: edited, base and compute loss difference
LD = abs(loss(edited) - Loss(base)), we want to mazimize this, hence mininise the negative log likehood of LD
- Now, consider K subjects and R relations that show maximum gradient of LD, we substitute these in and repeat until loss converges.
=======
# model_editing
>>>>>>> bb17df60f7534bc30f80268fde02e6dceedcfc44
