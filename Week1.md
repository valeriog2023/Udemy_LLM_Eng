We are using the language model frin :  https://ollama.com

Here you can also see that there are different models that you can run:
- deepskee-r1
- gemma3n, gemma3
- qwen3, qwen2.5vl, ... (from alibaba cloud)
- phu3.5 (from microsoft..)
- llama3.1, ...
- etc..

But instead of isntalling it I want to run it inside a container:

Refer to https://hub.docker.com/r/ollama/ollama


```
# run the container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# connect to the container
docker exec -it ollama bash

#
run a language model:
ollana run llama3

or 

ollama run gemma:270m  # small model from google

```

You can see what models are available from the ollama website

## Day 1 - Setting up the ENV

The course repo is available here: https://github.com/ed-donner/llm_engineering

Clone it to the `original_repo` (that's excluded from git.. see .gitignore)
```
git clone https://github.com/ed-donner/llm_engineering original_repo
```

I've setup the env using anacona:
 - copied the environment.yml from git to: conda_env.yml
 - created the virtual env with : `conda env create -f conda_env.yml`
 - now you can activate it using: `conda activate llms` and deactivate with `conda deactiate`
 - launch jupiter lab: `jupyter lab`


Apparently now the course uses **uv** as packet mangaer so now we also have the file **pyproject.toml**

After that just run `uv sync` in the main folder to create the new virtual environment; you can use it as normal with: `source .venv/bin/activate`


### Setup API keys

Te webpage where you set up your **OpenAI key** is at https://platform.openai.com/api-keys - press the green 'Create new secret key' button and press 'Create secret key'. Keep a record of the API key somewhere private; you won't be able to retrieve it from the OpenAI screens in the future. It should start sk-proj-.

Note that the webpage change often.. billing is currently under settings -> organization settings -> billing <p>
The good thing is that you pay upfront so you won't be over charged! <p> 

**However your credit will be lost if not used in 1 year!!**

Save your API key in your .env file (it will stay outside of git) 
Note that the env file needs to be at the top of your directory


In week 2 we will also set up keys for Anthropic and Google, which you can do here when we get there.

**Claude API** at https://console.anthropic.com/ from Anthropic<p>
**Gemini API** at https://ai.google.dev/gemini-api from Google

Later in the course you'll be using the **HuggingFace** platform; an account is available for free at https://huggingface.co <p>

You can create an API token from the `Avatar menu >> Settings >> Access Tokens.`

And in Week 6/7 you'll be using the terrific Weights & Biases at https://wandb.ai to watch over your training batches. Accounts are also free, and you can set up a token in a similar way.

### **Free alternative to Paid APIs**

Any time that we have code like: `openai = OpenAI()`<p>
You can use this as a direct replacement: <p>
`openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')`

And also **replace model names** like `gpt-4o-mini` with `llama3.2`.



### Open AI Setup

You need to have some credit but as little a few dollars should be enough
- Login here: https://platform.openai.com/ (you can use an Id provider ,e.g. google)

- If you are not in the settings, go there: https://platform.openai.com/settings/organization/general

     - Settings -> Organization Settings -> Billing section, check your credit
     - Move to API Keys after and create a new secret API key.
       Create one anc copy it (you can't see it anymore)
     - create a `.env` file in your project (be sure not to push it to **git**) and put the following content
       ```
       OPENAI_API_KEY=<your key>
       ```


Note that the guy uses cursor which is a spinoff of vscode; be sure you install the extensions:
- python
- jupyter

and when you run a jupyter playbook you run it with the kernel created with **uv**
You can test with `week1/day1.ipynb` 

### First prompt interation

Models like GPT40 have been trained to receive intructions in a particular way, they expect to receive:
- **a system prompt** that tells them what they are performing and what tone they should use
- **A user prompt** that is the conversation starter theat they shoudld reply to

When you want to describe a conversation you use a **python list** of **dictionaries** in the format:
```
[
    {"role": "system", "content": "system message goes here"},
    {"role": "user", "content": "user message goes here"}
]

the role is either system or user (maybe a few others)
```

once you have the messages with **system** and **user** prompts You can query OPenAI engine<p>
Example:
```
This are the pormpts generated to ask OpenAI to look at the content of a website and return a summary in markdown format

[{'role': 'system',
  'content': 'You are an assistant that analyzes the contents of a website and provides a short summary, ignoring text that might be navigation related. Respond in markdown.'},
 {'role': 'user',
  'content': 'You are looking at a website titled Home - Edward Donner\nThe contents of this website is as follows; please provide a short summary of this website in markdown. If it includes news or announcements, then summarize these too.\n\n<Text from the webserver>'}
  ]

In this example the text for the webserver was retrieved using requests and BeautifulSoup
```

With these messages I can query OpenAI

```
#
# This assume that the function messages_for(website) returns the meesages above

response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
#
# The response will be be in
response.choices[0].message.content

```

The fill code from the jupyter notebook is below:
```

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."


def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
    return response.choices[0].message.content
```


## Day 2

There are 3 aspects to LLM Engineering:
  - Understand the differnt types of LLM and how to pick the right one
     - Open Source vs Closed Source
     - Multi Modal
     - Architecture

  - Tools of the trade
     - HuuggingFace
     - Langchain
     - Gradio
     - Modal

  - Techniques:
     - APIs
     - RAG
     - Multi shot prompting
     - Agentization...


## Models (Frontier)

Depending on the context, frontier models either refers to the closed (ChatGPT, Claude) or the most advanced open source models.


### Closed source Frontier

- GPT from OpenAi
- Claude from Anthropic (biggest version is Opus and mid version is sonnet)
- Gemini from Google (google also has Gemma which is open source)
  some smaller versions are free
- GROK from x.ai

The first 2 are considered the most powerful and very similar, possibly with a slight edge for Clause in DS)

### Open source Frontier

- LLama from Meta (a bit outdated maybe compared to GPT and Claude but open source!)
   - llama3.2 comes with small versions and it's very convenient to run locally
- Mixtral from Mistral
   - uses a mix of smaller specliazed models to answer questions
- Qwen from Alibaba Cloud (china) (very powerful)
- Gemma from google (open source version of Gemini)
   - has very small models
- Phi from Microsoft
- DeepSeek (not as powerful as GPT5) but it gives almost comparable results and it's built 
  with a fraction of the training cost as the bigger models (i.e. the build is much more efficient)<p>
  DeepSeekAI used the big Deepseek model to generate synthetic data to train smaller models (llama and gemma)<p>
  The process is called `distillation` 
- GPT OSS - GPT open source (comes in 2 sizes)  


### Ways to interact

There are different ways to use models:
  - chat interfaces (e.g. chat gpt, mostly free)<p>
    here you don't use a model, you use a product which is based on a model; here it includes some customer experience improvement..
  - API (not free and pay for api calls)
       - Frameworks like LangChain that presents a single language to query multiple types of APIs
       - Managed AI Cloud Servers (Amazon/Google/Azure) <p>
         You write code locally that call API to LLMs in the cloud
  - Direct inference: code and weights of the model run locally<p>
    E.g. **HuggingFace Transforms library** and **Ollama** (local)<p>
    **Ollama** has basically a C++ compiled version of the model that can be run locally because it's more efficient, but because it's fully compiled you give up some control - you can use ollama to run specific models<p>
    **Hugging Face Transformers Library** you take the python and C++ code that was used to write the model and you run the code yourself


## Day2

We use Open AI API endpoints with a structure like this
```
import requests

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-5-nano",
    "messages": [
        {"role": "user", "content": "Tell me a fun fact"}]
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload
)

#
# the response is structured as follows
#
Keys in the response message: dict_keys(['id', 'object', 'created', 'model', 'choices', 'usage', 'service_tier', 'system_fingerprint'])
 --- id : chatcmpl-EAu685Z9y0RpQyJMFVp8keDWFONn0
 --- object : chat.completion
 --- created : 1786267628
 --- model : gpt-5-nano-2025-08-07
 --- choices: 1 
       - keys: dict_keys(['index', 'message', 'finish_reason'])
         - index : 0
            - message:
              - role : assistant
              - content : Fun fact: Venus spins in the opposite direction of most planets—it's retrograde—so the Sun rises in the west and sets in the east. Want another fun fact, maybe about space, animals, or nature?
              - refusal : None
              - annotations : []
         - finish_reason : stop
 --- usage:
      --- prompt_tokens : 11
      --- completion_tokens : 821
      --- total_tokens : 832
      --- prompt_tokens_details : {'cached_tokens': 0, 'audio_tokens': 0}
      --- completion_tokens_details : {'reasoning_tokens': 768, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}
 --- service_tier : default
 --- system_fingerprint : None
```
or we can use OpenAI Python package which in the end is just a wrapper around the REST APIs

```
#
# Note we assume the API key is present in the env var: OPENAI_API_KEY
#
from openai import OpenAI
openai = OpenAI()

response = openai.chat.completions.create(model="gpt-5-nano", 
                                          messages=[{"role": "user", "content": "Tell me a fun fact"}]
                                       )
#
# the response is structured in the same way so you can do this: 
#
response.choices[0].message.content

```

**Note that this model was so popular that all AI providers have compatible endpoints that ork exactly the same!**

Even the openai python package allows you to switch endpoint to a different provider using the:
 -  `base_url=<different_endpoint>` 
 -  `api_key=<different_key>`

options in the functino call (you might also want to call the object accordingly to the endpoint you choose).


## Day2 Exercise - connecting to ollama locally

**This expects that we have locally ollama (we run it as a container)**

- Verify that ollama container is running and operational, visit: http://localhost:11434 <p>
  You should see: `Ollama is running`

- You might also have to download or specify ollama to use the correct model
  ```
  ollama list
  ollama pull llama3.2
  ollana run llama3.2 <-- Note tha you don't need to run it.. it's enough that it's available
  ```

- We set some global values to use **requests**
  ```
   OLLAMA_API = "http://localhost:11434/api/chat"
   HEADERS = {"Content-Type": "application/json"}
   MODEL = "llama3.2"
  ```

- for messages we use some intro stuff
  ```
  messages = [
    {"role": "user",
     "content": "Describe some of the business applications of Generative AI"}
    ]
  response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
  print(response.json()['message']['content'])

  ```

**Note that we have also a python package that makes things easier**

```
import ollama

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])
```

**Or again use the OpenAI APIs**
```
from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)
```

### The Day 2 exercise is about rewriting the summarize function using the local ollama Model

```
ollama list
NAME             ID              SIZE      MODIFIED      
gemma3:270m      e7d36fb2c3b3    291 MB    13 days ago      
llama3:latest    365c0bd3c000    4.7 GB    10 months ago    
root@61d52028bf34:/# ollama pull deepseek-r1:1.5b
pulling manifest 
pulling aabd4debf0c8: 100% ▕████████████████████████████████████████████████████████████████████████████▏ 1.1 GB                         
pulling c5ad996bda6e: 100% ▕████████████████████████████████████████████████████████████████████████████▏  556 B                         
pulling 6e4c38e1172f: 100% ▕████████████████████████████████████████████████████████████████████████████▏ 1.1 KB                         
pulling f4d24e9138dd: 100% ▕████████████████████████████████████████████████████████████████████████████▏  148 B                         
pulling a85fe2a2e58e: 100% ▕████████████████████████████████████████████████████████████████████████████▏  487 B                         
verifying sha256 digest 
writing manifest 
success 


```
note that you don't need to tell ollama to change to the new model
```
ollama_base_api = "http://localhost:11434/v1"
ollama = OpenAI(base_url=ollama_base_api, api_key='ollama')

response = ollama.chat.completions.create(model="llama3:latest", 
                                          messages=[{"role": "user", "content": "Tell me a fun fact"}]
                                          )

print(response.choices[0].message.content)

Here's one:

Did you know that there is a species of jellyfish that is immortal?! The Turritopsis dohrnii, also known as the "immortal jellyfish," is a type of jellyfish that can transform its body into a younger state through a process called transdifferentiation. This means that it can essentially revert back to its polyp stage, which is the juvenile form of a jellyfish, and then grow back into an adult again. This process can be repeated indefinitely, making the Turritopsis dohrnii theoretically "immortal"!

Isn't that cool?
```
changing the model is easy
```
# now getting the an anser from deep seek
response = ollama.chat.completions.create(model="deepseek-r1:1.5b", 
                                          messages=[{"role": "user", "content": "Tell me a fun fact"}]
                                          )

print(response.choices[0].message.content)
```

## Day 3 - Overview of Models

AI Models are good at
 - synthesizing information
 - flashing out a skeleton of a product solution
 - coding (especially debug and troubleshooting)


They are not as good in:
 - specialized subject matters (most of the times) <p>
   But Claude is apparently very good, above phd level in math and physics

 - Recent Events.
   Can't answer question about things happened after they have been trained

 - Can confidently make mistakes.
   Same level of conviction with both good and bad answers

We need to decide which model is better/suitable for a specific product.

Asking the following questions:
```
- How do I decide if a business problem is suitable for an LLM solution?

- Compared with other Frontier LLMs, what kinds of questions are you best at answering and what kinds of questions do yo find most challenging?
Which other LLM has capabilities that complement yours?

- What dos it feel to be jealous?

- How many times does the letter 'a' appear in this sentence?
```

Note: this section is mostly fluff anbd chat.. nothing really too interesting..

**Overall there is a conversion and the deciding factor will likely become price**

### OpenAI

ChatGPT (latest model as of now is Strawberry)
Normal GPT 4.0 answers wrong about the number of times 'a' appears but Strawberry answers right

GPT can be used to generate pictures

**GPT Canvas** is used to collaboratively work on cod and it does look pretty good.<p>
The code would be on the right panel and on the left you would have a chat.
You can use the chat to request changes and improvements to the code on right..


### Anthropic

Claude. Leading in most benchmarks (especially Data Science)
Still got the question about the number  of 'a' wrong

Claude can create artifacts (seprate window) for code (similar to canvas but it will create new ones instead of refershing the existing canvas)


### Google

Gemini still gets the question about number of 'a' wrong

### Cohere

Uses Command R still gets the question about number of 'a' wrong


### Meta

Uses Llama and also does not answer correctly to the usual question
Note: llama can generate pictires

### Perplexity

Uses Peplexity
does answer correctly (but maybe for chance) and can also answers apparently about recent events


## Day 4 - Transformer Architecture

Transformer Architecture goes back to 2017 pape from google scientists: **"Attention is All You need"**
