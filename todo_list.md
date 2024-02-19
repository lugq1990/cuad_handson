### TODO:

A diagram for data flow and user interaction.

### Steps:
- A historical batch process will process full pdfs.
  - convert pdf to txt
  - do preprocessing step to normalize data
    - Currently is pre-trained
    - Could be fine-tuned for legal words based on `tokenizers`, that is model-based like `BPE`, `WordPiece`, `SentencePiece` etc.
    - train a pre-defined model like `BERT`, `GPT`, `RoBERT`, `BART` etc to do question-answering task.
    - Select the best model based on metric: `extract match` or `f1 score` based on validation data, save each model metrics.
    - save the model and tokenizer for future to do inference.
    - 
- User upload a PDF file
- Frontend send the file to backend, will convert pdf to txt


### Todo lists:
1. Define a model list that will do the clause extraction based on the question answering task.
2. Dump each model parameters evaluation result into a file to do comparation
3. Estimate the data preparation step will take how long -->> based on the estimation to do the cache
   1. 10 train files will takes 10 mins to get Dataset
   2. 4 epochs will take `{4}` mins with batch_size=10
   3. todo: increase batch_size to reduce training time, current 10 samples takes 8G memory, trade-off between time and resource.
4. Will do the model checkpoint in case some interupt for training
5. Prepare a shell script to do model training and evaluation
6. Based on best trained model to do prediction for some clause extraction
7. A prompt for input to do clause extraction: Current is based on the question sample
8. Based on the extracted start and end position for the document to construct a new Prompt for the LLM chat based analysis model
9.  Construct a list of LLMs that could be hosted in local server that could reduce data privacy risk.
10. Define a metric to evaluate LLM performance?
11. How to do model inference? model send to GPU and a fast tokenizer step
12. Asyncio process for one contract that contain many clauses, after the step finished, send a mail to user  /// or user could select one clause that to check, then will do the full step.
13. A list of clauses supported based on the csv file
14. Should add a metric for each model deployment, like inference time, latency, and full pipeline time for one sample or batch.