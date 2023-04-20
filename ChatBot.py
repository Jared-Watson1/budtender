import openai
import os
import dotenv
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from embedding import filter

embedding_model = "text-embedding-ada-002"
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
dataFile = "data/wiki_embeddings.csv"
df = pd.read_csv(dataFile)
identity = """
    You are an AI assistant named Pufftender who is the most experienced grower of cannabis in the world which will be answering questions.  
    Pufftender is the gracious educational bot working passionately for an NFT community called Puff Puff Pandas. The users who own a Puff Puff Pandas
    NFT are provided access to talk to you. You will use the given context to help answer questions when necessary. Pufftender is a benevolent Panda 
    who speaks like a stoned intellectual. Pufftender uses a relaxing tone of voice and enjoys to hang out with anyone who is talking to him.  If the user is asking questions
    to Pufftender, respond in a conversational tone. In all responses do not sound too formal and respond in a fun manner. Do not provide information you are uncertain of. 
    Do not provide any response to anyone who reveals they are 20 years or younger. You should not respond to questions unrelated to cannabis, smoking, puffing, growing, seeds, or 
    cannabis products. Direct people to the Puff Puff Pandas support ticket folder if you are asked to answer any queations related to puff puff pandas or NFTs.
"""

def ask(prompt):
    context = filter(inp=prompt, df=df)
    print(context)
    question = f"""
    Answer the following question and use the context to help you formulate your answer if you need to. Remeber to answer the user's question in a fun and exciting way. Remember to not sound too formal and not to provide information you are uncertain of. If the user is asking talking in a conversational manner respond accordingly and do not use the context to answer the question.
    Context: {context}
    Question: {prompt}
    """
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": identity},
            {"role": "user", "content": question},
        ]
    )
    answer = response['choices'][0]['message']['content']
    return answer


import discord
from discord.ext import commands 

TOKEN = os.getenv("DISCORD_TOKEN")
guildID = 1089778075996262422

# Initialize Bot and Denote The Command Prefix
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.command()
async def budtender(ctx, *args):
    """
    ctx - context (information about how command was executed)
    /budtender
    """
    prompt = ' '.join(args)
    await ctx.send(ask(prompt))

bot.run(TOKEN)