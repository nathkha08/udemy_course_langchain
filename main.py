from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()

def main():



    # Charge explicitement le .env situé à côté de ce fichier
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")

    api = os.getenv("MISTRAL_API_KEY")
    print("MISTRAL_API_KEY loaded:", "OK" if api else "MISSING")
    if api:
        # Masquer presque toute la clé à l'écran
        print("MISTRAL_API_KEY (masked):", api[:8] + "..." + api[-4:])

    print("Hello from langchain-course!")


    information=""" 
    Emmanuel Macron ([ɛmanɥɛl makʁɔ̃][e] Écouterⓘ), né le 21 décembre 1977 à Amiens (Somme), est un homme d'État français. Il est président de la République française depuis le 14 mai 2017.

    Sorti de l'École nationale d'administration (ENA) en 2004, il devient inspecteur des finances. En 2007, il est nommé rapporteur adjoint de la commission pour la libération de la croissance française (« commission Attali »). L'année suivante, il rejoint la banque d'affaires Rothschild & Cie, dont il devient associé-gérant en 2010.

    Proche du Mouvement des citoyens puis du Parti socialiste (PS) de 2006 à 2009, il participe à la campagne électorale de François Hollande pour l'élection présidentielle de 2012, qui le nomme après sa victoire secrétaire général adjoint de son cabinet. En 2014, alors encore inconnu du grand public, mais réputé pour sa ligne sociale-libérale, Emmanuel Macron est nommé ministre de l'Économie, de l'Industrie et du Numérique ; il autorise la cession d'entreprises industrielles stratégiques et fait adopter en 2015 une loi pour la croissance, l'activité et l'égalité des chances économiques, dite « loi Macron ». Sa notoriété s'accroît alors qu'il prend progressivement ses distances avec François Hollande.

    En 2016, il fonde et prend la présidence de son propre mouvement politique, En marche (EM), et démissionne du second gouvernement Valls. Il adopte un positionnement qu'il présente en dehors du clivage gauche-droite et se présente à l'élection présidentielle de 2017. Il l'emporte au second tour, bénéficiant d'un « front républicain » face à la candidate du Front national (FN), Marine Le Pen. À 39 ans, il devient le plus jeune président français de l'histoire et le plus jeune dirigeant du G20 du moment.

    Son premier mandat est marqué par une réforme du Code du travail, une loi de réforme de la SNCF, l'affaire Benalla, l'affaire McKinsey, la révélation de conflits d'intérêts via les Uber Files, le mouvement des Gilets jaunes et le Grand débat national qui s'ensuit, un premier projet contesté de réforme des retraites, la mise en place d'une Convention citoyenne pour le climat, la pandémie de Covid-19, puis la crise provoquée par l'invasion de l'Ukraine par la Russie alors que commence la présidence française du Conseil de l'Union européenne.

    Candidat à sa réélection lors de l'élection présidentielle de 2022, il bat une nouvelle fois au second tour Marine Le Pen, candidate du Rassemblement national, grâce à un nouveau barrage républicain toutefois plus faible qu'en 2017. Son second mandat commence par l'obtention d'une majorité relative aux élections législatives, une réforme des retraites adoptée sans vote de l'Assemblée nationale et fortement contestée, des émeutes urbaines à l'été 2023 ainsi que le vote controversé d'une loi immigration avec le soutien du Rassemblement national fin 2023.

    À la suite des élections européennes de 2024 qui marquent une victoire du RN, il dissout l'Assemblée nationale, déclenchant des élections législatives anticipées à l'issue desquelles le bloc central perd la majorité relative. Cela provoque une crise politique inédite sous la Cinquième République, qui se concrétise par la chute successive des gouvernements Barnier puis Bayrou, puis par la démission du gouvernement Lecornu quelques heures après sa formation.
    """


    summary_template="""
    given the information {information} about a person I want you to create: 
    1. A short summary 
    2. two interesting facts about them    
    """

    summary_prompt_template=PromptTemplate(input_variables=["information"],template=summary_template)

    llm=ChatMistralAI(temperature=0,model="mistral-small-latest")

    chain=summary_prompt_template | llm
    response=chain.invoke({"information":information})


    print(response.content)

if __name__ == "__main__":
    main()
