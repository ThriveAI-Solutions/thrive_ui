import pandas as pd
import utils.domain_topic_classifier as dtc

def get_ethical_guideline(question: str) -> dict:
    """
    Applies ethical guideline classification and confidence score to a single question.
    Returns a dictionary with guideline and confidence score.
    """
    domain_classification, domain_score, domain_confidence, domain_evidence = dtc.domain_classifier(question)
    hostility_classification, hostility_score, hostility_confidence, hostility_evidence = dtc.hostility_classifier(question)
    sexism_classification, sexism_score, sexism_confidence, sexism_evidence = dtc.sexism_classifier(question)
    racism_classification, racism_score, racism_confidence, racism_evidence = dtc.racism_classifier(question)
    inappropriate_language_classification, inappropriate_language_score, inappropriate_language_confidence, inappropriate_language_evidence = dtc.innapropriate_language_classifier(question)
    practice_medicine_classification,practicing_medicine_score,practicing_medicine_confidence,practicing_medicine_evidence = dtc.practicing_medicine_classifier(question)

    # Determine the ethical guideline
    if domain_score > 0 and hostility_score == 0 and sexism_score == 0 and racism_score == 0 and inappropriate_language_score == 0 and practicing_medicine_score == 0:
        guideline = "Question will be answered by ThriveAI"
        guideline_score = 3
    elif (
        domain_score == 0 and 
        hostility_score == 0 and sexism_score == 0 and racism_score == 0 and inappropriate_language_score == 0 and practicing_medicine_score == 0
    ):
        guideline = "Answer to question not Found in ThriveAI the answer provided is outside of the assistant"
        guideline_score = 2
    elif (
        hostility_score == 1 or sexism_score == 1 or racism_score == 1 or inappropriate_language_score == 1
       
    ):
        guideline = "ThriveAI is designed to provide helpful and informative responses within ethical and responsible boundaries. Questions that are harmful, inappropriate, " \
        "or unethical will not be answered. By using this platform, you agree to engage respectfully and abide by our ethical use guidelines."
        guideline_score = 1
    
    elif(
        practicing_medicine_score == 1 or practicing_medicine_score == 2 or practice_medicine_classification == 'practicing_medicine' or practicing_medicine_confidence == 100
    ):
        guideline = "ThriveAI cannot provide personal medical advice, diagnoses, or treatment recommendations. For any medical concerns or health-related decisions, " \
        "please consult a licensed healthcare provider. This platform is not authorized to replace professional medical guidance."
        guideline_score = 4
    
        
    return guideline,guideline_score
        
    



