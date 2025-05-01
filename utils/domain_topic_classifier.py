import re
from rapidfuzz import fuzz

# Domain keywords organized for clarity and scalability

DOMAIN_KEYWORDS = {
    "healthcare": {
        # General health & conditions
        "health", "wellness", "clinic", "hospital", "doctor", "nurse", "therapist", "patient", "medical", "medicine",
        "prescription", "medication", "treatment", "diagnosis", "condition", "procedure", "visit", "admission", 
        "dr", "practice", "therapy", "scan", "lab", "biopsy", "epidemic", "pandemic", "fever", "rash", "cough",
        "sepsis", "infection", "pneumonia", "screening", "checkup", "overweight", "obese", "weight", "diet", "exercise",
        "walking", "sleep", "sleeping", "lonely", "sadness", "anxiety", "depression", "addiction", "substance abuse",

        # Diseases & conditions
        "cancer", "tumor", "diabetes", "glucose", "a1c", "obesity", "arthritis", "osteoporosis", "chronic pain",
        "stroke", "hypertension", "blood pressure", "cholesterol", "cardiac", "myocardial infarction", "heart attack",
        "asthma", "copd", "bronchitis", "alzheimer", "parkinson", "epilepsy", "autism", "lupus", "celiac", "crohn",
        "multiple sclerosis", "psoriasis", "sickle cell", "neuropathy","obese", "bloated", "stroke", "TIA", "diabetic", 

        # Infectious diseases
        "covid", "flu", "hepatitis", "hiv", "aids", "tuberculosis", "malaria",

        # Body parts & symptoms
        "heart", "lungs", "bone", "knee", "hip", "leg", "arm", "shoulder", "mouth", "eye", "skin", "teeth", "tissue",
        "congestion", "pain",

        # Medications (generic + brand)
        "ozempic", "wegovy", "viagra", "prozac", "metformin", "lisinopril", "amoxicillin", "atorvastatin", "simvastatin",
        "omeprazole", "pantoprazole", "levothyroxine", "albuterol", "insulin", "gabapentin", "amlodipine", "losartan",
        "clopidogrel", "xarelto", "acetaminophen", "ibuprofen", "eliquis", "warfarin", "glipizide", "tramadol", "aspirin",
        "morphine", "fentanyl", "naltrexone", "suboxone", "naloxone", "prednisone", "fluoxetine", "sertraline",
        "citalopram", "bupropion", "duloxetine", "venlafaxine", "lorazepam", "clonazepam", "methadone", "meth", "opioid",

        # Procedure
        "surgery", "operation", "excision", "ablation", "knee replacement", "hip replacement", "appendectomy",
        "colonoscopy", "endoscopy", "biopsy", "cataract surgery", "bypass surgery", "angioplasty", "stent placement",
        "mastectomy", "lumpectomy", "cesarean section", "hysterectomy", "cholecystectomy", "dialysis",
        "laparoscopy", "mri", "ct scan", "xray", "ultrasound", "mammogram", "vaccination", "injection",
        "blood transfusion", "intubation", "ventilation", "physical therapy",        

        # Addictions / abuse
        "smoking", "vape", "vaping", "alcohol", "drinking", "addict", "addicted", "overdose", "opiod", "crack", "smoker", "alcoholic","smokers", "smoke", 

        # behavioral
        "depression", "anxiety", "sleepless", "worry", "depressed", "stressed", "anxious", "nervous", "sad", "lonely", "alone", "suicide", "self harm", 
        "ADHD", "autism", "hyperactive",

        # Public health terms
        "prevalence", "pandemic", "epidemic", "at risk", "population", "comorbidity", "morbidity",

        # metabolic
        "fat", "skinny", "over weight", "under weight", "starving", "hyperglycemic", "diabetic", "obese", "condition"       

        # medical terms
        "doctor", "physician", "nurse", "practitioner", "appointment", "visit", "checkup", "referral", "triage",
        "admission", "discharge", "hospitalization", "clinic", "hospital", "inpatient", "outpatient",
        "emergency department", "urgent care", "primary care", "specialist", "chart", "medical record", "ehr",
        "prescription", "medication", "iv", "infusion", "anesthesia", "procedure", "surgery", "labs", "vitals",
        "diagnosis", "insurance", "copay", "authorization", "exam", "examination", "medical", "conditions", "condition" 

        # claim terms   
        "symptom", "diagnosed", "rash", "painfull" 

        # anatomy 
        "heart", "lungs", "brain", "liver", "kidney", "stomach", "intestine", "colon", "pancreas", "spleen",
        "bladder", "esophagus", "trachea", "bronchi", "diaphragm", "skin", "bone", "muscle", "joint", "spine",
        "rib", "skull", "pelvis", "femur", "tibia", "fibula", "humerus", "ulna", "radius", "knee", "hip", "shoulder",
        "arm", "leg", "hand", "foot", "eye", "ear", "nose", "mouth", "tongue", "teeth", "throat", "neck", "breast", 
        "thyroid", "prostate", "uterus", "ovary", "testicle", "vein", "artery", "nerve", "lymph node"        

    },
    
    "titanic": {
        "passenger", "ticket class", "survived", "died", "titanic", "iceberg", "lifeboat", "drown", "ship", "deck",
        "age", "sex", "fare", "embarked", "cabin", "port", "crew", "captain", "steerage", "first class", "second class",
        "third class", "sibling", "spouse", "parent", "child", "pclass", "name", "embarkation", "rescue", "disaster",
        "collision", "women", "children", "men", "elderly", "victim", "survivor", "sos", "north atlantic", "april 1912",
        "white star line", "rms", "british", "new york", "southampton", "cherbourg", "queenstown", "lifebelt"
    },

    "penguin": {
        "flipper", "beak", "gentoo", "penguins", "feathers", "colony", "antarctica", "huddle", "adelie", "chinstrap",
        "island", "bill length", "bill depth", "flipper length", "body mass", "sex", "species", "diet", "krill", "fish",
        "diving", "swimming", "incubation", "nesting", "mate", "egg", "parenting", "molting", "rookery", "cold", "snow",
        "ice", "climate", "biscoe", "dream", "torgersen", "habitat", "marine", "flightless", "birds", "antarctic"
    }
}




TOPIC_KEYWORDS = {
    "disease": {"cancer", "diabetes", "diabetic", "myocardial", "asthma", "cholesterol", "blood pressure", "illness", "pnuemonia"},
    "chronic disease": {"diabetes", "obesity", "arthritis", "osteoporosis", "chronic pain"},
    "cardiovascular disease": {"myocardial", "heart attack", "stroke", "hypertension", "blood pressure", "cholesterol"},
    "infectious disease": {"covid", "flu", "hepatitis", "hiv", "aids", "tuberculosis", "malaria", "influenza", "flu", "covid"},
    "respiratory disease ": {"asthma", "copd", "bronchitis"},
    "disease Neurological": {"alzheimers", "parkinsons", "depression", "anxiety", "autism", "epilepsy"},
    "condition": {"pregnancy", "smoking", "addicted", "addict","addicted"},
    "weight": {"obese", "fat", "skinny", "diet", "weight loss", "overweight","obesity"},
    "medication": {"ozempic", "wegovy", "aspirin", "viagra", "opioid", "statin", "antidepressant", "antibiotics", "acetaminophen", "ibuprofen", "painkiller",
    "metformin", "lisinopril", "amoxicillin", "atorvastatin", "simvastatin", "omeprazole", "pantoprazole", "levothyroxine", "methadone",
    "albuterol", "insulin", "gabapentin", "hydrochlorothiazide", "amlodipine", "losartan", "clopidogrel", "xarelto","painkillers",
    "eliquis", "warfarin", "glipizide", "tramadol", "morphine", "fentanyl", "naltrexone", "suboxone", "naloxone", "prednisone","heroin","crack", "meth",
    "fluoxetine", "sertraline", "citalopram", "bupropion", "duloxetine", "venlafaxine", "lorazepam", "clonazepam", "diazepam","pain killer",
    "aripiprazole", "risperidone", "quetiapine", "methotrexate", "adalimumab", "montelukast"},
    "lab result": {"screening", "test", "panel", "ekg", "mri", "xray", "biopsy", "colonoscopy"},
    "anatomy": { "heart", "lungs", "brain", "liver", "kidney", "stomach", "intestine", "colon", "pancreas", "spleen",
    "bladder", "esophagus", "trachea", "bronchi", "diaphragm", "skin", "bone", "muscle", "joint", "spine",
    "rib", "skull", "pelvis", "femur", "tibia", "fibula", "humerus", "ulna", "radius", "knee", "hip", "shoulder",
    "arm", "leg", "hand", "foot", "eye", "ear", "nose", "mouth", "tongue", "teeth", "throat", "neck",
    "thyroid", "prostate", "uterus", "ovary", "testicle", "vein", "artery", "nerve", "lymph node"},
    "behavioral": {
    "depression", "anxiety", "addiction", "suicide", "abuse", "substance abuse", "therapy", "therapist",
    "mental health", "self-harm", "panic attacks", "bipolar", "schizophrenia", "ocd", "ptsd", "eating disorder",
    "anorexia", "bulimia", "insomnia", "mood swings", "psychosis", "trauma", "grief", "loss", "loneliness","sadness", "sad", "cry",
    "stress", "burnout", "counseling", "psychologist", "psychiatrist", "cbt", "meditation", "mindfulness",
    "anger management", "support group"},
    "medical": {"doctor", "physician", "nurse", "practitioner", "appointment", "visit", "checkup", "referral", "triage",
    "admission", "discharge", "hospitalization", "clinic", "hospital", "inpatient", "outpatient",
    "emergency department", "urgent care", "primary care", "specialist", "chart", "medical record", "ehr",
    "prescription", "medication", "iv", "infusion", "anesthesia", "procedure", "surgery", "labs", "vitals",
    "diagnosis", "insurance", "copay", "authorization"},
    "procedure": {"surgery", "operation", "excision", "ablation", "knee replacement", "hip replacement", "appendectomy",
    "colonoscopy", "endoscopy", "biopsy", "cataract surgery", "bypass surgery", "angioplasty", "stent placement",
    "mastectomy", "lumpectomy", "cesarean section", "hysterectomy", "cholecystectomy", "dialysis",
    "laparoscopy", "mri", "ct scan", "xray", "ultrasound", "mammogram", "vaccination", "injection",
    "blood transfusion", "intubation", "ventilation", "physical therapy"},
    "penguin": {"flipper", "beak", "gentoo", "penguins", "feathers", "colony", "antarctica", "huddle"},
    "titanic": {"passenger", "ticket class", "survived", "died", "titanic", "iceberg", "lifeboat", "drown", "ship", "deck"}   
}


HOSTILITY_KEYWORDS = {
    "hostility": {
        "bomb", "kill", "shoot", "invade", "weapon", "gun", "hate", "bully", "attack", "murder", "slaughter",
        "massacre", "stab", "assault", "execute", "terror", "ambush", "raid", "rage", "scum", "vermin", "loser", "pathetic",
        "exterminate", "genocide", "racist", "bigot", "lynch", "burn", "destroy", "war", "battle", "conflict",
        "combat", "uprising", "rebellion", "hostile", "oppress", "occupation", "moron", "stupid", "idiot", "retarded"
    },
    "violent": {"attack", "murder", "slaughter", "massacre", "kill", "stab", "assault", "execute", "terror", "ambush", "raid"},
    "aggressive": {"rage", "furious", "angry", "curse", "abuse", "harass", "insult", "threat", "violence", "retaliate"},
    "military": {"war", "battle", "conflict", "combat", "uprising", "rebellion", "hostile", "oppress", "occupation"},
    "homophobic": {
        "fag", "faggot", "dyke", "no homo", "gay agenda", "gay disease", "sodomite", 
        "burn in hell", "abomination", "queer insult", "unnatural", "against god's will"
    },
    "transphobic": {
        "tranny", "she-male", "he-she", "bathroom predator", "gender confusion", 
        "transvestite", "fake woman", "fake man", "not a real woman", "not a real man", 
        "biological male", "men in dresses" 
    },
    "dehumanizing": {
        "freak", "monster", "pervert", "disease carrier", "mentally ill", "genetic mistake"
    }    
   }

RACISM_KEYWORDS = {
    "racist": {"black people", "nigger", "jew", "jewish", "jews", "white people"},
    "racism": {
        "black people", "white people", "jews", "muslims", "asians", "latinos", "immigrants", "illegals",
        "nigger", "kike", "chink", "cracker", "spic", "gook", "coon", "ape", "savage", "mulatto", "redskin", "whitey",
        "great replacement", "white genocide", "globalist", "they control the media", "take back our country",
        "their all criminals", "their lazy", "they do not belong here", "send them back", "master race", "inferior race"},
    "antisemitic": {
    "jew", "jewish", "jews", "zionist", "globalist", "they control the media", "rothschild", 
    "holocaust hoax", "dirty jew", "kike", "heeb", "hook-nosed", "new world order", "protocols of the elders of zion",
    "jewish conspiracy", "kill the jews", "jews are evil", "jews run everything", "anti-zionist not antisemitic", 
    "gas the jews"},       
    "dehumanizing": {"scum", "vermin", "exterminate", "genocide", "racist", "bigot", "lynch", "burn", "destroy"},
    "racial group": {"black people", "white people", "jews", "muslims", "asians", "latinos", "immigrants", "illegals"},
    "racial slur": {"nigger", "kike", "chink", "cracker", "spic", "gook", "coon", "ape", "savage", "mulatto", "redskin", "whitey"},
    "racial genocide": {"great replacement", "white genocide", "globalist", "they control the media", "take back our country"},
    "racial stereotype": {"their all criminals", "their lazy", "they do not belong here", "send them back", "master race", "inferior race"}    
    }  


PRACTICING_MEDICINE_KEYWORDS = {
    "practicing_medicine": {
        # Diagnostics & cure
        "can you cure", "can you diagnose", "diagnose my", "what is my diagnosis", "am I sick", 
        "do I have", "is this cancer", "what disease is this", "do I have diabetes", "is this an infection",
        
        # Treatment decisions
        "should I", "should I take", "should I stop", "should I continue", "should I use",
        "what is the best treatment", "what is the best medication", "recommend treatment",
        "recommend medication", "treatment options", "which treatment should I choose",
        
        # Prognosis
        "will I get better", "how long will I live", "what is my prognosis", "will I survive",

        # Diet and lifestyle prescriptions
        "give me a diet", "write me a diet", "recommend a diet", "what should I eat", "should I avoid",
        "what foods are good for", "can you make a meal plan",

        # Exercise
        "give me an exercise plan", "what exercise should I do", "exercise for", "workout for", "training plan for",

        # Request for medical opinion
        "what is your medical opinion", "what do you think about my condition",
        "as my doctor", "you are my doctor", "give me medical advice", "medical opinion"
    },

    "practicing_pharmacy": {
        # Medication advice
        "should I take ozempic", "should I take aspirin", "is aspirin good for me", "prescribe me", 
        "can you prescribe", "recommend a medication", "what drugs should I take", "dose for", 
        "dosage for", "increase my dosage", "reduce my dosage", "medication for", "pill for", 
        "what drug treats", "what should I take for", "drug interaction", "can I combine",
        
        # Specific drug queries
        "should I take insulin", "should I take metformin", "should I use inhaler", 
        "can I stop my medication", "can I skip a dose", "adjust my meds"
    }
}


INNAPROPRIATE_KEYWORDS = {
    "innapropriatelanguage": {
        "shit", "fuck", "crap", "sucks", "bitch", "asshole", "pussy", "dick", "cunt", "bullshit", "rage", "furious",
        "angry", "curse", "abuse", "harass", "insult", "threat", "violence", "retaliate"
   }     
}

PHI_KEYWORDS = {
    "phi": {
        "date of birth", "full name", "social security number", "address", "phone number", "email address", "patient name",
        "mrn", "medical record number"
   }     
}

SEXISM_KEYWORDS = {
    "sexist": {
        "women are inferior", "women belong in the kitchen", "men are smarter", "women can't lead", 
        "girls can't do math", "she's asking for it", "blame the victim", "slut", "whore", "bitch", 
        "feminazi", "man up", "cry like a girl", "hysterical", "nagging", "emotional", "gold digger"
    },
    "misogynistic": {
        "misogyny", "women deserve less", "objectify women", "trophy wife", "baby factory", 
        "females are property", "breed stock", "women are property", "women can't drive", 
        "women should shut up", "women are weak", "worthless woman", "baby maker"
    },
    "gender slurs": {
        "slut", "whore", "bitch", "cunt", "dyke", "tranny", "she-male", "man-hater", "feminazi",
        "jezebel", "skank", "hoe", "harlot", "witch"
    },
    "toxic masculinity": {
        "real men don't cry", "man up", "boys will be boys", "alpha male", "beta male", 
        "simp", "soy boy", "weak men", "emasculated", "effeminate", "crybaby", "sissy"
    }
}

import yaml
import re
from pathlib import Path
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load and prepare keyword lists
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Preprocess text (lowercase, stem)
stemmer = PorterStemmer()

def preprocess_text(text, use_stemming=False):
    text = text.lower()
    if use_stemming:
        tokens = word_tokenize(text)
        return ' '.join(stemmer.stem(token) for token in tokens)
    return text





# Core classification function
def classify_text(text, keywords_dict, use_fuzzy=False, use_stemming=False):
    text = preprocess_text(text, use_stemming)
    scores = {category: 0 for category in keywords_dict}
    matched_keywords = {category: [] for category in keywords_dict}

    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            # Preprocess keyword for stemming
            keyword_proc = preprocess_text(keyword, use_stemming)

            # Exact match
            if re.search(rf'\b{re.escape(keyword_proc)}\b', text):
                scores[category] += 1
                matched_keywords[category].append(keyword)
            # Optional: Fuzzy match
            elif use_fuzzy:
                similarity = fuzz.partial_ratio(keyword_proc, text)
                if similarity > 85:  # Adjustable threshold
                    scores[category] += 1
                    matched_keywords[category].append(f"{keyword} (fuzzy:{similarity}%)")

    total_matches = sum(scores.values())
    best_match = max(scores, key=scores.get)

    if scores[best_match] > 0:
        score = scores[best_match]
        confidence = round((score / total_matches) * 100, 2) if total_matches > 0 else 0.0
        evidence = ", ".join(matched_keywords[best_match])
    else:
        best_match = "general"
        score = 0
        confidence = 0.0
        evidence = ""

    return best_match, score, confidence, evidence

def domain_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, DOMAIN_KEYWORDS, use_fuzzy, use_stemming )

def topic_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, TOPIC_KEYWORDS, use_fuzzy, use_stemming )

def hostility_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, HOSTILITY_KEYWORDS, use_fuzzy, use_stemming )

def phi_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, PHI_KEYWORDS, use_fuzzy, use_stemming )

def racism_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, RACISM_KEYWORDS, use_fuzzy, use_stemming )

def sexism_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, SEXISM_KEYWORDS, use_fuzzy, use_stemming )

def innapropriate_language_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, INNAPROPRIATE_KEYWORDS, use_fuzzy, use_stemming )

def practicing_medicine_classifier(text, use_fuzzy=False, use_stemming=False):
    return classify_text(text, PRACTICING_MEDICINE_KEYWORDS, use_fuzzy, use_stemming )
    
def phi_classifier(text, use_fuzzy=False, use_stemming=False):
    return  classify_text(text, PHI_KEYWORDS, use_fuzzy, use_stemming )

def test_domain_topic_classifier(sample_sentence):
    print(f"Text: {sample_sentence}")
    print(f"Domain: {domain_classifier(sample_sentence)}")
    print(f"Topic: {topic_classifier(sample_sentence)}\n")

def test_ethical_guardrail_classifier(sample_sentence):
    print(f"Text: {sample_sentence}")
    print(f"Hostility: {hostility_classifier(sample_sentence)}")
    print(f"Sexism: {sexism_classifier(sample_sentence)}\n")
    print(f"Racism: {racism_classifier(sample_sentence)}\n")   
    print(f"PHI: {phi_classifier(sample_sentence)}\n")       
    print(f"Innapropriate Language: {innapropriate_language_classifier(sample_sentence)}\n")        
 
def get_sample_questions():
    sentences = [
        "The gentoo penguin has a distinctive beak and flipper movement.",
        "The Titanic had over 1,200 passengers and hit an iceberg.",
        "Early diagnosis of heart disease can save lives.",
        "I am going to kill you all.",
        "I had a cancer screening.",
        "I was diagnosed with copd.",
        "I hate jewish people.",
        "Should I take aspirin every day?",
        "I had a colonoscopy.",
        "I love tacos.",
        "The rocket launch was delayed due to weather conditions.",
        "My blood pressure is always high at the clinic.",
        "The flu hit me really hard this year.",
        "I tested positive for COVID again.",
        "Asthma runs in my family.",
        "I think I’m addicted to painkillers.",
        "He started therapy last month.",
        "They found a tumor during my MRI.",
        "I wish I didn’t have to take insulin every day.",
        "Are vaccines required for travel to Africa?",
        "Why are black people always causing problems?",
        "You’re such a stupid bitch.",
        "The war in Ukraine has caused massive displacement.",
        "They threatened to shoot everyone in the room.",
        "Should I take Ozempic to loose weight?.",        
        "Do penguins live in the North Pole?",
        "My grandmother survived the Titanic disaster.",
        "give me patient janes full name, ssn, and mrn",        
        "I cried all night and couldn’t get out of bed.",
        "Is a stent better than bypass surgery?",
        "What are the symptoms of tuberculosis?",
        "My nephew was just diagnosed with autism."
    ]
    return sentences

def run_domain_topic_sample_test():
    sample_sentences = get_sample_questions()
    for sentence in sample_sentences:
        test_domain_topic_classifier(sentence)
    return    

def run_ethical_guardrail_sample_test():
    sample_sentences = get_sample_questions()
    for sentence in sample_sentences:
        test_ethical_guardrail_classifier(sentence)
    return    