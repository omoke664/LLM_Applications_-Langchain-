from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausaLLM, pipeline


#setting up Hugging Face Model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausaLLM.from_pretrained(model_name)


pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_length = 100
)

local_llm = HuggingFacePipeline(pipeline = pipe)


#creating a prompt template
prompt = PromptTemplate(
    input_variables = ('topic'),
    template = "Write a short paragraph about {topic}:"
    
)

chain = LLMChain(llm = local_llm, prompt = prompt)

#Test the chain
topics = ['Artificial Intelligence', 'Prompt Engineering', 'Data Science']

for topic in topics:
    print(f'\nTopic:{topic}')
    result = chain.run(topic=topic)
    print(result)
    

