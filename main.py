from agent import *

if __name__ == "__main__":
    sujet_recherche = "Intelligence artificielle"
    question = "Quelles sont les applications principales de l'intelligence artificielle dans la vie quotidienne?"
    
    print("\nQuestion:", question)
    print("\nRecherche d'informations sur:", sujet_recherche)
    print("\nRéponse:")
    
    response = wikipedia_agent(question, sujet_recherche)
    print(response)