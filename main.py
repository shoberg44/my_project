from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, render_template
import logging
import re

logging.basicConfig(level=logging.DEBUG)
VERSION = "llama3.2"

app = Flask(__name__)

def init_ollama():
    try:
        create_prompt = ChatPromptTemplate(
            [
                ("system", "You are my personal assistant"),
                ("user", "Question: {question}"),
            ]
        )
    
        llama_model = OllamaLLM(model = VERSION)
        format_output = StrOutputParser()

        chatbot_pipeline = create_prompt | llama_model | format_output
        return chatbot_pipeline
    
    except Exception as e:
        logging.error(f"Error initializing Ollama: {e}")
        raise

def format_output(text):
    return re.sub(r'', '', text)

chatbot_pipeline = init_ollama()

@app.route("/", methods=["GET", "POST"])
def main():
    query_input = None
    output = None

    if request.method == "POST":
        query_input = request.form.get("query_input")
        
        if query_input:
            try:
                response = chatbot_pipeline.invoke({'question': query_input})
                output = format_output(response)
                print(response)
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                raise
    
    return render_template("ai_chatbot.html", output=output)

if __name__ == "__main__":
   app.run(debug=True)

