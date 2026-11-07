# Week3 - OPEN SOURCE MODELS 

## HUGGING FACE

There are 2 parts to higging face:
- **hub/platform**: This is the web site and mainly it is used because it has:
   - Source models (milions and more).. it's like a repo for DS models. 
   - Data sets. It has taken the place of kaggle for datasets (you can find a lot of finance datasets)
   - Spaces: hosted apps, like gradio apps (use deploy option) that you can share and use.  
     This is anything that can run in a container

**You need an account on Hugging Face (free)**

- **libraries**: code/open source libraries that everyone uses it. They implement foo instance **transformers** (mostly in pytorch)
  and you can download them, tweak the weights and run the models yourself

  Note that this is different from ollama because ollama is an actual product, packaged and made efficient (also pre-baked)
  Here you can have access to the actual code, so you can change it and fine tuning..<p>
  In particular we use 6 libraries here:
   - **hub**: let you connect to the platform (hub above) to get models and datsets
   - **datasets**: this let you manipulate datasets in an efficient way
   - **transformers**: it's the library that let you work with models, change them, run them, etc..
   - **peft**: performance efficient parameter tuning library
   - **trl**: transformer reinforcement library
   - **accelerate**: distribute a library across multiple GPUs


## GOOGLE COLAB

Link to the Course Colab Page: <p>
https://colab.research.google.com/drive/1DjcrYDZldAXKJ08x1uYIVCtItoLPk1Wr?usp=sharing


Link to empty/new Colab Notebook:<p>
https://colab.research.google.com


Google Colab
 - allows you to run a notebook in your browser window on a Google owned computer out there in the cloud.
 - allows you to easily collaborate and share the notebook with other people.
 - ingtegrate wioth other google services (like gooogle drive) which is very convenient
 - **It gives you access to high end GPUs** In particular, you can choose runtime which are the environments you use for your notebook:
    - CPU based
    - GPU based; The lower spec/cost is T4(15GB GPU RAM) and should be free but you can pay to get higher level GPUs

## Basic usage    
Link to notebook test: [LLM_week3_day1](https://colab.research.google.com/drive/1cIeoaUlVBgo_QQCS_Z2sni7ceHa2JwYP); it's not shared.

Basic: 
 - **connect**: (button on the top right) select the running environment and change runtime env.
   Here you can select the **T4 GPU** and in the drop down menu (once connected) you can:
     - see the resources used (menu: view resource)
     - disconnect and delete run time (also available from the run time menu)
 - **share**: (button on the top right) select to share
 - **runtime menu**: use for restart sessions etc.. (if you disconect, you'll have to reinstall packages)<p>
   and because it's in cloud, you'll have to build your environment every time.

Note that sometime, google will change your running environment and move you to CPU or other types that might not work for you. Keep that in mind.

Also remember to save before leaving Colab or you will loose your work.


## Check the GPU

```
# Let's check the GPU - it should be a Tesla T4

# this is actually calling the command nvidia-smi and getting the output
gpu_info = !nvidia-smi

# basic checks here
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
  if gpu_info.find('Tesla T4') >= 0:
    print("Success - Connected to a T4")
  else:
    print("NOT CONNECTED TO A T4")


### expected result when connected on colab/T4
-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
Success - Connected to a T4

```


## Connect to Colab account

From you hugging face account:
 - go to your avatar -> profile -> API Access Tokens
 - select Write ( we need write acces)
 - give the token a name: (**UDEMY_LLM_GOOGLE_COLAB_TOKEN**) and click on create

Now move to Colab:
 - select the key icon on the left (secrets)
 - Add a new secret: 
    - Name: HT_TOKEN
    - Value: hf_<...>
    - select the enable flag on the left

  Secrets will not be shared if you share a colab notebook.


In the Colab notebook you would have something like this:
```
#
#  CONNECTING TO HUGGING FACE
#  This uses the hugging face hub library
#  If the cell runs, it's all good
#
from huggingface_hub import login
from google.colab import userdata


hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

Then you can call a Diffusion pipeline (what is it exactly?) from hugging face for instance to generate a picture
```
from IPython.display import display
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompt = "A class of data scientists learning AI engineering in a vibrant high-energy pop-art style"

image = pipe(prompt=prompt, num_inference_steps=30).images[0]

display(image)
``` 

You can also update/upgrade Hugging Face libraries
```
!uv pip install --upgrade datasets==3.6.0
```

and generate audio (before it was a picture)
```
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from IPython.display import Audio

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')
embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})

Audio(speech["audio"], rate=speech["sampling_rate"])
```


# HUGGING FACE LIBRARIES

## HIGH LEVEL CODE: PIPELINES
Simpler interface to the lower level code: higher level APIs to carry out standard tasks quickly
e.g. the model has already been train and you just want to run it in common inference use cases:
 - sentiment analysis
 - classifier
 - named entity recognition tasks
 - question answering
 - summarizing
 - translation

It is very easy to create a pipeline.. the basic structure is:
```
my_pipeline = pipeline("the_task_I_want_to_do")

Followed by

result = my_pipeline(my_input)
```

Note that:
  - pipeline tasks are **inference** tasks, i.e. the model has already been trained and now you use it.
    **training** tasks are not done with pipelines.
  - you might want to cehck that you are running on a GPU, you can get the gpu (nvidia) info as follows
    ```
    gpu_info = !nvidia-smi
    gpu_info 
    ``` 
  - import and how to connect.. note that you need the env var **HF_TOKEN** set (with a write token)
    ```
    import torch
    from google.colab import userdata
    from huggingface_hub import login
    from transformers import pipeline
    from diffusers import DiffusionPipeline  # ysed for pictures
    from datasets import load_dataset
    import soundfile as sf
    from IPython.display import Audio
    
    my_pipeline = pipeline(task,      # required.. what you want the pipeline to do (string/enum)
                           model=xx,  # optional, you can specify a model but it's not required
                                      #    it will default to one model and tell you
                           device=xx  # optional, you can specify if you want to use GPU 
                                      #    for nvidia, it's 'cuda'
                                      #    for apple/mac silicon (if you test locally), it's 'mps'
                           )
    #
    # then you ask to infer something  (e.g. below)                     
    ``` 


### PIPELINES EXAMPLES

#### Sentiment Analysis (task: sentiment-analysis )

Below some exapmles for sentiment analysis..<p>
The lesson that we learn is that the model we pick makes a difference.. we should check what model hugging face makes available for a specific task and select the correct one.

```
my_simple_sentiment_analyzer = pipeline("sentiment-analysis", device="cuda")
result = my_simple_sentiment_analyzer("I'm super excited to be on the way to LLM mastery!")
print(result)


print("--------")
# this is not a very positive example
result = my_simple_sentiment_analyzer("I should be more excited to be on the way to LLM mastery!")
print(result)
```
Result would be something like this:
```
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
config.json: 100%
 629/629 [00:00<00:00, 20.5kB/s]
model.safetensors: 100%
 268M/268M [00:08<00:00, 29.1MB/s]
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 2.85kB/s]
vocab.txt: 
 232k/? [00:00<00:00, 10.1MB/s]
Device set to use cuda
[{'label': 'POSITIVE', 'score': 0.9993460774421692}]
-------------
[{'label': 'POSITIVE', 'score': 0.9008274674415588}]  # <- note that this is misclassified.. it's not that positive
```

We can now test with a model that is not default but specialized for sentiment analysis

```
better_sentiment = pipeline("sentiment-analysis", 
                 model="nlptown/bert-base-multilingual-uncased-sentiment",  # model dedicated to sentiment analysis
                 device="cuda")
result = better_sentiment("I should be more excited to be on the way to LLM mastery!!")
print(result)
```

Result is
```
# Result is in starts (here we get 3 out of 5 so it definitely gets something)
[{'label': '3 stars', 'score': 0.3901219367980957}]
```

#### Named Entity Recognition (task: ner)

It is used to categorize items in a text.. (e.g. person names, entities, etc..). it works token by token
and returns something like the following
```
ner = pipeline("ner", device="cuda")
result = ner("AI Engineers are learning about the amazing pipelines from HuggingFace in Google Colab from Ed Donner")
for entity in result:
  print(entity)

...
Device set to use cuda
{'entity': 'I-ORG', 'score': np.float32(0.999476), 'index': 1, 'word': 'AI', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': np.float32(0.9962089), 'index': 2, 'word': 'Engineers', 'start': 3, 'end': 12}
{'entity': 'I-ORG', 'score': np.float32(0.8205678), 'index': 11, 'word': 'Hu', 'start': 59, 'end': 61}
{'entity': 'I-ORG', 'score': np.float32(0.651493), 'index': 12, 'word': '##gging', 'start': 61, 'end': 66}
{'entity': 'I-ORG', 'score': np.float32(0.960993), 'index': 13, 'word': '##F', 'start': 66, 'end': 67}
{'entity': 'I-ORG', 'score': np.float32(0.92517126), 'index': 14, 'word': '##ace', 'start': 67, 'end': 70}
{'entity': 'I-MISC', 'score': np.float32(0.8882749), 'index': 16, 'word': 'Google', 'start': 74, 'end': 80}
{'entity': 'I-MISC', 'score': np.float32(0.67307675), 'index': 17, 'word': 'Cola', 'start': 81, 'end': 85}
{'entity': 'I-PER', 'score': np.float32(0.9989441), 'index': 20, 'word': 'Ed', 'start': 92, 'end': 94}
{'entity': 'I-PER', 'score': np.float32(0.99872154), 'index': 21, 'word': 'Don', 'start': 95, 'end': 98}
{'entity': 'I-PER', 'score': np.float32(0.96185666), 'index': 22, 'word': '##ner', 'start': 98, 'end': 101}  
```


#### Other tasks:
Just specify one of the other tasks in the pipeline, an appropriate model and device=`cuda` (if you use a GPU)

 - **question-answering**<p>: This requires as param when called to inproduce an inference: 
     - question
     - context: The context (you could generate context from the result of another pipelibe)
 - **summarization**:  this takes as parameter:
     - text
     - max_length, min_length 
     - do_sample
   Results are in `result[0]['summary_text']`  
 - **translation_<from>_to_<to>**, e.g. **translation_en_to_it**
   Results are in `result[0]['translation_text']`
 - **text_generation**
   Results are in `result[0]['generated_text']`
 - **zero-shot-classification**: puts a sentence in a bucket with no previous examples. 
   Takes as input:
   - text: <just text>
   - candidate_labels: list of possible labels we want to use to possibly classify the test
   It returns the probability of the text being any of the possible labels given
 - **image/audio generation**  

There are actually many other pipelines, so have a look at the higging face docs:

https://huggingface.co/docs/transformers/main_classes/pipelines

There's also this list of Tasks for Diffusion models instead of Transformers, following the image generation example where I use DiffusionPipeline above.
https://huggingface.co/docs/diffusers/en/api/pipelines/overview



## LOWER LEVEL CODE: TOKENIZERS AND MODELS
They provide more power and control

 - [Day1 (personal) colab notebook](https://colab.research.google.com/drive/1cIeoaUlVBgo_QQCS_Z2sni7ceHa2JwYP#scrollTo=5iiiD5GFvwXA)
 - [Day2 (personal) colab notebook][https://colab.research.google.com/drive/1m23CbF5wVXMhL4W6eiMlmmbXAhAw2Mck#scrollTo=jhLouIeW5kXw]
