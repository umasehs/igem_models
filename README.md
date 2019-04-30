# Umaseh's stuff

I have two trained models here so far.

1. activator_model predicts whether CRP will act as an activator or as a repressor on some promoter.
2. bindingsite_model predicts where the CRP-binding site is within a given sequence.

Look at ```train.py``` in each folder to see the important code. Go to ```test/test.py``` in the activator_model folder to just start using it. The classifier itself is saved as ```clf.joblib``` in each folder.

## More Details

activator_model uses 10 hand-selected features from the 390ish features provided by PredCRP. On the most recent version of RegulonDB, activator_model is much more accurate than PredCRP (__activator_model achieves 95.2% training accuracy, 93.7% validation accuracy, and 100% test accuracy on the 23 sequences__, while PredCRP achieves ~87% on the newest RegulonDB and 22/23 on the test sequences).

bindingsite_model uses just the genetic sequences. It trains on half the data, and is validated using the other half. First, it splits each 42-length sequence into every possible 22-length sequence (because these are the lengths recommended in the PredCRP paper -- I'm guessing most CRP-binding sites are 22 bases long). It takes these 22-length sequences (22-grams) and one-hot encodes the bases. Then, it trains to identify whether each 22-gram (now with 22*4 = 88 features) is a binding site or not. To validate, we perform the same preprocessing on the validation sequences, then we use the classifier to predict the probability that each 22-gram is the binding site, and we see whether the highest-probability binding site in each sequence is the actual binding site. __bindingsite_model has 96.8% training accuracy, and 90.4% validation accuracy,__ which is pretty good considering the proportional size of our validation set (usually models use far less of the data for validation, because they need as much as possible for training).

## What's with the '# %%' symbols everywhere in the code

If you use Visual Studio Code, you can run the script like a Jupyter notebook, where each cell is denoted by ```# %%```.

## What do I need to install first

I only tested this with Python 3.6 on Linux, but it should theoretically work on any OS as long as you have Python 3. I'm too lazy to list out all the prerequisite packages you need to install, but you can find them in the import statements at the top of each script, and then get them using conda or pip. Just use the most recent version of each one, and message me if you have any questions/issues/feedback.
