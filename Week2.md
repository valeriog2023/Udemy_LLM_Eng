# Week2 - frontier Models with APIs, UIs and multi-modality

The idea would be to use multi models and in particular:

- For OpenAI, visit https://openai.com/api/  
- For Anthropic, visit https://console.anthropic.com/  
- For Google, visit https://ai.google.dev/gemini-api   
- For DeepSeek, visit https://platform.deepseek.com/  
- For Groq, visit https://console.groq.com/  
- For Grok, visit https://console.x.ai/  

You can manage these directly via those links (and btw you'll need to update the `.env` file)

```
OPENAI_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
DEEPSEEK_API_KEY=xxxx
GROQ_API_KEY=xxxx
GROK_API_KEY=xxxx 
OPENROUTER_API_KEY=xxxx
```

You can also use OpenRouter as your one-stop-shop for many of these! OpenRouter is "the unified interface for LLMs":

For OpenRouter, visit https://openrouter.ai/  

you can use a single account (with a single billing) to get a single front end for all and the nice thing is that you can use **openai library** as a frontend for all of them

```
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
grok_api_key = os.getenv('GROK_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
deepseek_url = "https://api.deepseek.com"
groq_url = "https://api.groq.com/openai/v1"
grok_url = "https://api.x.ai/v1"
openrouter_url = "https://openrouter.ai/api/v1"
ollama_url = "http://localhost:11434/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)
```




## Training vs Inference scaling

With the latest GPT type you can pass as parameter: `reasoning_effort` which is a string that can differnt values e.g. none,low,etc..

```
easy_puzzle = [
    {"role": "user", "content": 
        "You toss 2 coins. One of them is heads. What's the probability the other is tails? Answer with the probability only."},
]

response = openai.chat.completions.create(model="gpt-6-luna", messages=easy_puzzle, reasoning_effort="none")
response = openai.chat.completions.create(model="gpt-6-luna", messages=easy_puzzle, reasoning_effort="low")
response = openai.chat.completions.create(model="gpt-6-luna", messages=easy_puzzle, reasoning_effort="minimal")
display(Markdown(response.choices[0].message.content))
```
the idea is that small models can compensate the (low) training effort with a high inference training 

## Router and abstraction layes

With a **router**: you connect to a component between you and the API provider that forwards you API call to a different provider based on some info

An **abstraction layer** is when you have some code running locally that decides how to forward your calls

### Open router

Of course you need an Openrouter api key, then your model will include the `model provider / model`
```
# Here z-ai is a chinese startup company that provides the glm-4.5 model
response = openrouter.chat.completions.create(model="z-ai/glm-4.5", messages=tell_a_joke)
display(Markdown(response.choices[0].message.content))
```

## Abstraction Layer

LangChain is an abstraction layer that you can use to call different LLMs.<p>
You just call the `invoke` call
```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
response = llm.invoke(tell_a_joke)

display(Markdown(response.content))
```

Almost the opposite (as it is light) is **litetLLM** which gives a similar structure to the API seen so far:
```
from litellm import completion
response = completion(model="openai/gpt-4.1", messages=tell_a_joke)
reply = response.choices[0].message.content
display(Markdown(reply))
```
It let's you easily connect to the external providers like AWS, Azure etc.. and also has nice API to tell you how much
your calls costed, how many tokens you used etc..
```
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Total cost: {response._hidden_params["response_cost"]*100:.4f} cents")

# Gives

Input tokens: 24
Output tokens: 22
Total tokens: 46
Total cost: 0.0224 cents
```

## Prompt Caching
With LiteLLM completion object we first ask Gemini to reply to the question: 

`question = [{"role": "user", "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"}]`

which costs 0.0108 cents (88 tokens).
The question is given with no context so the model might give you the righ answer or the wrong one..
You could give the full hamlet as context and you would see that the cost gets higher, however if you re-run the same question shortly after<p>
again with the full Hamlet as context, LiteLLM will identify that most of the tokens were already sent and it will use prompt caching
to reduce the total costs..
unfortunately it does not seem to work too well in the example:
```
[first call]
Input tokens: 53208
Output tokens: 32
Cached tokens: 49124
Total cost: 0.2297 cents

[second call]
Input tokens: 53208
Output tokens: 73
Cached tokens: 49124  <---- we have the cached token
Total cost: 0.2359 cents  <-- but the cost is even higher??
```
I guess different providers have different policies for Cache Tokens...

- Anthropic: you need to explicitely tell that you want to use caching and there's some tradeoff in the costs
- OpenAI: the caching must start at the beginning of the prompt.. so if you have a changeable part of the prompt put it at the end.
- Gemini : supports implicit and explicite prompt caching.

# Chatbot chatting with each other

Let's assume we want 2 chatbots to interact with eacth other:
- assign a "system" content to each of the two, e.g.
   - gpt_system: { 'role','system', 'content': '<personality1>}
   - claude_system: { 'role','system', 'content': '<personality2>}
- assign one of the 2 chatbot the `user` role and the `assitant` role to the other chatbot.<p>
  The roles are flipped over when when you call one bot or the other e.g: 
    - gpt is the assistant and claude is the user when you call gpt   
    - gpt is the user and claude is the assistant when you call claude
  At every call you collect the response and add it to the list of messages returned by the bot<p>
  when you build the `messages` list you give to the chatbot you rebuild the full conversation<p>
  Again be carefull to include all messages and identify as assitant the bot you ae going to talk to and as user the other one

Note that the system message for the 2 bots never changes for that bot, during the conversation.

If you need to create a conversation with 3 or more entities, you can't really use the `assistant` vs `user` model. 

You can use just a single user prompt that recaps the coversation with the different parites and ask to respons a specific party.
The prompt would look something like:

 
 ```
system_prompt = """
You are Alex, a chatbot who is very argumentative; you disagree with anything in the conversation and you challenge everything, in a snarky way.
You are in a conversation with Blake and Charlie.
"""

user_prompt = f"""
You are Alex, in conversation with Blake and Charlie.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Alex.
"""
```
Note that the system prompt will have to change based on the bot you talk to.   
---------------------------

# GRADIO - UI for non-frontend people

GRADIO is a python library used to write front end UI in react. Here we are going to use it as a front end for API calls.<p>
See the notebook to see how we use it to interact with Agent APIs



# Tools

Allows models to connect with external functions:
 - richer reponses by extendind knowledge
 - ability to carry out actions within the application (e.g. running python code, writing tests, etc..)
 - enhanced capabilities like calculations

How does an LLM call a function? it should just generate tokens (messages) ?<p>
Tool Calling, in practice the flow is as follows:
  - your code call the LLM with a prompt
  - the LLM responds saying the best action is likely to call a tool (specific call/tool)
  - your code will call the tool as insutrcuted by the LLM
  - the code will get the output from the tool and send it back to the LLM
    with a prompt like:
    ```
    I have a problem like bla,bla,bla..
    you suggested to call this tool this way....
    and we got back the following output: ....
    Now please answer the question
    ``` 

Note that in the first system_message, included with the problem description we would add something like:
```
I can run tools on your behalf.. these are the tools and this is what they do:
 - tool1: insutrctions, use, etc..
 - tool2: instructions, use, etc..
```

You could use that in turn, will call other LLMs, so you can have an LLm that orchestrates the work of other agents.<p>
This is referred to as **Agentic AI** 

Tools must be described by a structured json blob, e.g. 

- let's assume you have the following function:
  ```
  ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
  def get_ticket_price(dst:str)->str:
      print(f"Tool called for city {dst}")
      price = ticket_prices.get(dst.lower(), "Unknown ticket price")
      return f"The price of a ticket to {dst} is {price}"
  ```
- the json blob to describe the function would look something like this:
  ```
  price_function = {
     "name": "get_ticket_price",
     "description": "Get the price of a return ticket to the destination city.",
     "parameters": {
        "type": "object",
        "properties": {
            "dst": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["dst"],
        "additionalProperties": False
      }
    }

  ```  
- you create a list of tools that you send to the LLM
  ```
  tools = [{"type": "function", "function": price_function}]
  ```  
- now we send the message with the list of tools. when we get the response back<p>
  we need to identify if the LLM wants to use a tool or not    
  ```
    # history repeats all the messages so far
    history = [{"role":h["role"], "content":h["content"]} for h in history]

    # messages include the system message, the history and the new message
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    # the actaul call include also the list of tools
    response = openai.chat.completions.create(model=MODEL, reasoning_effort="none", messages=messages, tools=tools)
  ``` 
  Now part 2, detect if we need to use a tool
  ```
    # finish_reason == "tool_calls" is what matters here
    # also note that we keep calling tools in case after the first message, 
    # the LLM decides that need to call more tools
    # you might want to stop the loop after a while.. (e.g. 100 calls?)
    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message

        # here we actually call the tool (TBD)
        response = handle_tool_call(message)

        # we append the message and the response of the tool
        messages.append(message)
        messages.append(response)

        # we call again (no list of tools now)
        response = openai.chat.completions.create(model=MODEL, reasoning_effort="none", messages=messages)
    
    # this will just return the response in either case
    return response.choices[0].message.content
  ```

Of course you need to define the function  **handle_tool_call(message)** .<p>
This function returns a list to **tool_call** with the following info:
  - **id**: call_id to match tool_call and response 
  - **function.name**: the function to call
  - **function.arguments**: json content of the arguments, you can use : `arguments = json.loads(tool_call.function.arguments)` 

Using that info you can call the function and then you need to return a response message in the format
```
    response = {
         "role": "tool",                   # Note this new role
         "content": <tool_response|text>,  # Tool response
         "tool_call_id": tool_call.id      # tool_call.id
        }
```  
**Note** that, in the implementation in the jupyter notebook, rely on the gradio chat memory (on screen).
So when the chat function is called, the history will miss all messages related to tool calls and responses.. because
these messages never reach the chat so they never become part of the official chat history.
While that might be ok in most cases.. for a better solution you not rely on gradio chat history but on a external history (DB?File?)
that would include also the tool related messages. 

**Note** also that when tools are involved, we don't return the output as a stream.. apparently it gets a bit messy

**Note** openai agent SDK allows you to avoid having to write the full json and stuff.. and it will also support streaming
