from huggingface_hub import login
import os
from typing import List

import torch
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#################
#    CONFIG     #
#################
huggingface_token = os.environ.get("HUGGINGFACE_API_KEY") 
login(token=huggingface_token)

# Configuration pour le chargement quantifié du modèle
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Chargement du modèle et du tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Configuration des embeddings de HuggingFace
embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Template pour le prompt RAG
template = """
Tu es un assistant IA qui utilise Mistral-7B pour répondre à des questions.
Tu disposes des informations suivantes extraites de Wikipedia :

{context}

En utilisant ces informations et tes connaissances, réponds à la question suivante :
{question}

Si tu ne connais pas la réponse ou si les informations fournies ne sont pas suffisantes, indique-le clairement.
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)




# Fonction pour générer une réponse avec Mistral
def generate_mistral_response(prompt):
    # Vérifier si le prompt est un objet PromptValue
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            encoded_input,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extraction de la réponse générée par le modèle
    response_parts = decoded_output.split("user\n")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    else:
        return decoded_output  # Retourne la sortie complète si le splitting échoue
    
# Fonction pour rechercher des informations sur Wikipedia et préparer la base de connaissances
def create_wikipedia_knowledge_base(query, lang="fr", load_max_docs=2):
    # Chargement des documents depuis Wikipedia
    print(f"Recherche d'informations sur Wikipedia pour: {query}")
    loader = WikipediaLoader(
        query=query,
        load_max_docs=load_max_docs,
        lang=lang
    )
    documents = loader.load()
    
    if not documents:
        print("Aucun document trouvé sur Wikipedia.")
        return None
    
    # Découpage des documents en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Création de la base de connaissances vectorielle
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Fonction principale de l'agent
def wikipedia_agent(query, research_topic, lang="fr"):
    # Récupération des documents pertinents de Wikipedia
    retriever = create_wikipedia_knowledge_base(research_topic, lang=lang)
    
    if not retriever:
        return "Je n'ai pas pu trouver d'informations sur ce sujet sur Wikipedia."
    
    # Création du pipeline RAG
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Création du pipeline RAG avec gestion correcte des types
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
    )
    
    # Obtenir le prompt formaté
    formatted_prompt = rag_chain.invoke(query)
    
    # Utiliser le prompt formaté pour générer une réponse
    response = generate_mistral_response(formatted_prompt)
    
    return response