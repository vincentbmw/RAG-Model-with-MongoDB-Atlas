# Import libraries
import os, sys
import pymongo
sys.path.insert(0, '../')
from urllib.request import urlopen
from dotenv import find_dotenv, dotenv_values
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
#from llama_index.llms.llama_api import LlamaAPI
#from llama_index.llms.gemini import Gemini
#from llama_index.llms.replicate import Replicate
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline
from llama_index.core import Settings

# Define variables
DB_NAME = 'dogs'
COLLECTION_NAME = 'type'
config = dotenv_values(find_dotenv())
service_context = None
vector_store = None
storage_context = None
index= None

# Initialization function
def initialize():
    ATLAS_URI = config.get('ATLAS_URI')
    print(f"ATLAS_URI detected is: {ATLAS_URI}")

    if not ATLAS_URI:
        raise Exception ("'ATLAS_URI' is not set. Please set it in .env before continuing...")

    os.environ['LLAMA_INDEX_CACHE_DIR'] = os.path.join(os.path.abspath('../'), 'cache')

    mongodb_client = pymongo.MongoClient(ATLAS_URI)
    print ('Atlas client succesfully initialized!')
    return mongodb_client

# LLM Function
def setup_llm():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = config.get('HUGGING_FACE_API_KEY')
    os.environ["REPLICATE_API_TOKEN"] = config.get('REPLICATE_API')
    GOOGLE_API_KEY = config.get('GOOGLE_API_KEY')
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    #Settings.llm = Replicate(
    #    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5", 
    #    system_prompt="""You are an efficient language model designed to respond promptly to user inquiries.
    #        Responses should be concise and to the point, avoiding unnecessary elaboration unless requested by the user. But if user don't like the dog breed that you gived to user, just give another dog breeds""", 
    #    temperature=0.75,
    #    length_penalty=1,
    #    max_new_tokens=1000
    #)

    model_id = 'infgrad/stella-base-en-v2'# go for a smaller model if you dont have the VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SentenceTransformer("infgrad/stella-base-en-v2")

    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=1200
    )

    Settings.llm = HuggingFacePipeline(pipeline=pipe)
    resp = Settings.llm.complete("Paul Graham is ")
    print(resp)
    service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model, llm=Settings.llm)

#Connect LLM to MongoDB
def connect_llm(client):
    global index
    vector_store = MongoDBAtlasVectorSearch(mongodb_client = client,
                                 db_name = DB_NAME, collection_name = COLLECTION_NAME,
                                 index_name  = 'idx_embedding',
                                 )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# Query function
def run_query(question):
    response = index.as_query_engine().query(question)
    return response

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_text = """
    🐾 Welcome to Pawspective! 🐾
    
    To begin, please describe the type of dog you're interested in. For example, you can tell me about the size, temperament, or any specific traits you prefer in a dog.
    
    Let's find your perfect furry companion! 🐶🐾
    """
    await update.message.reply_text(start_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
    🐾 Welcome to Pawspective! 🐾
    
    To use this bot, simply follow these steps:
    1. Type /start to begin.
    2. Ask any questions about dogs that you want to know.
    3. Explore the suggested dog breeds to find your ideal furry companion!
    
    Feel free to ask any questions about dog breeds or how to use the bot. 🐶🐾
    """
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    response_text = run_query(user_message)
    await update.message.reply_text(response_text.response)
    
def main():
    print('Starting the Bot...')
    app = Application.builder().token(config.get('BOT_API')).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    print('Polling...')
    app.run_polling(poll_interval=3)

if __name__ == '__main__':
    ip = urlopen('https://api.ipify.org').read()
    print (f"My public IP is '{ip}.  Make sure this IP is allowed to connect to cloud Atlas")
    mongodb_client = initialize()
    setup_llm()
    connect_llm(mongodb_client)
    main()