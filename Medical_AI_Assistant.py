"""
Fixed RAG-based Medical AI Assistant using Gemini API
Uses urllib instead of requests to avoid conflicts.
"""

import streamlit as st
import os
import urllib.request
import urllib.parse
import json
from typing import Dict, List, Any


class FixedRAGGeminiMedicalChatbot:
    """Fixed RAG-based medical chatbot using Gemini API with urllib."""
    
    def __init__(self):
        """Initialize the fixed RAG Gemini medical chatbot."""
        self.disclaimer = "⚠️ **IMPORTANT DISCLAIMER**: I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
        
        # Your Gemini API key - Replace with your actual API key
        self.api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBlrgNbZbb8-i0b52gvtnlssDmtiI1ftwY")
        self.model_name = "gemini-1.5-flash"
        self.available_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-1.5-flash-001"]
        
        # Medical knowledge base for RAG
        self.medical_knowledge = {
            "headache": {
                "info": "Headache relief: rest in dark room, cold compress, hydration, gentle neck stretches, OTC pain relievers. See doctor for severe or frequent headaches.",
                "sources": "headache_guide"
            },
            "fever": {
                "info": "Fever treatment: rest, hydration, cool compresses, fever reducers, monitor temperature. Seek medical attention if fever is high or persistent.",
                "sources": "fever_guide"
            },
            "stomach ache": {
                "info": "Stomach relief: bland foods, clear liquids, avoid spicy foods, peppermint tea, gentle heat. Seek medical attention if pain is severe.",
                "sources": "digestive_guide"
            },
            "period pain": {
                "info": "Period pain relief: heat therapy, gentle exercise, OTC pain relievers, relaxation techniques, avoid caffeine/alcohol. Consult doctor if severe.",
                "sources": "women_health_guide"
            },
            "knee pain": {
                "info": "Knee pain relief: rest, ice/heat therapy, gentle stretching, supportive footwear, avoid high-impact activities. Consult doctor for persistent pain.",
                "sources": "joint_health_guide"
            },
            "back pain": {
                "info": "Back pain relief: rest, ice/heat therapy, gentle stretching, good posture, firm mattress, avoid heavy lifting. Consult doctor for persistent pain.",
                "sources": "back_health_guide"
            },
            "cold": {
                "info": "Cold treatment: rest, hydration, saline nasal spray, humidifier, honey (adults), 7-10 days recovery. See doctor if symptoms persist.",
                "sources": "respiratory_health_guide"
            },
            "cough": {
                "info": "Cough relief: warm liquids, humidifier, honey (adults), salt water gargle, throat lozenges, avoid irritants. See doctor if cough persists.",
                "sources": "respiratory_health_guide"
            },
            "sore throat": {
                "info": "Sore throat relief: salt water gargle, warm liquids, throat lozenges, rest voice, avoid irritants. See doctor if symptoms persist.",
                "sources": "throat_health_guide"
            },
            "nausea": {
                "info": "Nausea relief: small bland meals, clear liquids, avoid strong smells, ginger tea, comfortable position. Seek medical attention if severe.",
                "sources": "digestive_health_guide"
            },
            "fatigue": {
                "info": "Fatigue management: adequate sleep, regular schedule, hydration, balanced meals, exercise, stress management. Consult doctor if persistent.",
                "sources": "energy_health_guide"
            },
            "stress": {
                "info": "Stress management: deep breathing, meditation, exercise, sleep, limit caffeine, talk to others, professional counseling if needed.",
                "sources": "mental_health_guide"
            },
            "insomnia": {
                "info": "Sleep improvement: regular schedule, comfortable environment, avoid screens/caffeine, relaxation techniques. Consult doctor for persistent problems.",
                "sources": "sleep_health_guide"
            },
            "anxiety": {
                "info": "Anxiety management: deep breathing, meditation, exercise, sleep, limit caffeine, talk to others, professional counseling if needed.",
                "sources": "mental_health_guide"
            },
            "vitamin b12": {
                "info": "Vitamin B12: essential for nerve function, found in animal products, deficiency causes fatigue/weakness. Consult doctor for testing and supplementation.",
                "sources": "nutrition_guide"
            },
            "dehydration": {
                "info": "Dehydration prevention: drink water, eat water-rich foods, avoid excess caffeine/alcohol, watch for thirst/dark urine. Seek medical attention if severe.",
                "sources": "hydration_guide"
            }
        }
        
        # Medical keywords - Comprehensive list
        self.medical_keywords = [
            # General Health Terms
            'health', 'medical', 'medicine', 'healthcare', 'wellness', 'fitness',
            'disease', 'disorder', 'condition', 'syndrome', 'illness', 'sickness',
            'symptom', 'symptoms', 'sign', 'signs', 'indication', 'manifestation',
            'treatment', 'therapy', 'cure', 'healing', 'recovery', 'rehabilitation',
            'diagnosis', 'diagnose', 'examination', 'checkup', 'screening', 'test',
            'prevention', 'preventive', 'prophylaxis', 'immunization', 'vaccination',
            'first aid', 'emergency', 'urgent', 'critical', 'acute', 'chronic',
            
            # Body Systems & Organs
            'heart', 'cardiac', 'cardiovascular', 'circulation', 'blood', 'artery',
            'vein', 'pulse', 'heartbeat', 'chest', 'chest pain', 'angina',
            'lung', 'lungs', 'respiratory', 'breathing', 'breath', 'airway',
            'stomach', 'gastric', 'digestive', 'gastrointestinal', 'intestine',
            'liver', 'kidney', 'kidneys', 'bladder', 'urinary', 'urine',
            'brain', 'neurological', 'nervous system', 'nerve', 'nerves',
            'muscle', 'muscles', 'muscular', 'tendon', 'tendons', 'ligament',
            'bone', 'bones', 'skeletal', 'spine', 'spinal', 'joint', 'joints',
            'skin', 'dermatological', 'hair', 'nails', 'teeth', 'dental',
            'eye', 'eyes', 'vision', 'visual', 'ear', 'ears', 'hearing',
            'nose', 'nasal', 'throat', 'pharyngeal', 'mouth', 'oral',
            
            # Common Symptoms
            'pain', 'ache', 'aching', 'sore', 'soreness', 'tender', 'tenderness',
            'fever', 'temperature', 'hot', 'chills', 'sweating', 'sweat',
            'headache', 'migraine', 'dizziness', 'vertigo', 'lightheaded',
            'nausea', 'nauseous', 'vomiting', 'vomit', 'throwing up',
            'diarrhea', 'loose stools', 'constipation', 'bowel movement',
            'fatigue', 'tired', 'exhausted', 'weak', 'weakness', 'lethargy',
            'sleep', 'sleeping', 'insomnia', 'sleepless', 'drowsy', 'drowsiness',
            'appetite', 'hunger', 'thirst', 'dehydration', 'dehydrated',
            'swelling', 'swollen', 'inflammation', 'inflamed', 'redness',
            'rash', 'itchy', 'itching', 'burning', 'stinging', 'numbness',
            'tingling', 'cramps', 'cramping', 'spasms', 'stiffness', 'rigidity',
            'shortness of breath', 'wheezing', 'coughing', 'cough', 'sneezing',
            'runny nose', 'congestion', 'stuffy nose', 'sore throat',
            'chest pain', 'abdominal pain', 'stomach pain', 'belly pain',
            'back pain', 'neck pain', 'shoulder pain', 'knee pain', 'joint pain',
            
            # Medical Conditions & Diseases
            'diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin',
            'hypertension', 'high blood pressure', 'low blood pressure',
            'heart disease', 'heart attack', 'stroke', 'cardiac arrest',
            'asthma', 'bronchitis', 'pneumonia', 'tuberculosis', 'covid',
            'cancer', 'tumor', 'tumors', 'malignant', 'benign', 'metastasis',
            'arthritis', 'rheumatoid', 'osteoarthritis', 'gout', 'fibromyalgia',
            'depression', 'anxiety', 'panic', 'panic attack', 'mental health',
            'bipolar', 'schizophrenia', 'ptsd', 'trauma', 'stress', 'stressed',
            'allergy', 'allergic', 'anaphylaxis', 'hay fever', 'asthma',
            'infection', 'bacterial', 'viral', 'fungal', 'sepsis', 'fever',
            'influenza', 'flu', 'cold', 'common cold', 'sinusitis', 'bronchitis',
            'pneumonia', 'tuberculosis', 'hepatitis', 'hiv', 'aids',
            'migraine', 'epilepsy', 'seizure', 'seizures', 'parkinson',
            'alzheimer', 'dementia', 'memory loss', 'confusion',
            'anemia', 'iron deficiency', 'vitamin deficiency', 'malnutrition',
            'obesity', 'overweight', 'underweight', 'eating disorder',
            'thyroid', 'hypothyroidism', 'hyperthyroidism', 'hormone',
            'kidney disease', 'liver disease', 'cirrhosis', 'hepatitis',
            'ulcer', 'ulcers', 'gastritis', 'acid reflux', 'heartburn',
            'ibs', 'crohn', 'colitis', 'diverticulitis', 'hemorrhoids',
            
            # Medications & Treatments
            'medicine', 'medication', 'drug', 'drugs', 'pill', 'pills',
            'tablet', 'capsule', 'injection', 'inject', 'shot', 'shots',
            'antibiotic', 'antibiotics', 'painkiller', 'painkillers',
            'antidepressant', 'antidepressants', 'anxiety medication',
            'blood pressure medication', 'diabetes medication', 'insulin',
            'chemotherapy', 'radiation', 'surgery', 'operation', 'procedure',
            'therapy', 'physical therapy', 'occupational therapy',
            'psychotherapy', 'counseling', 'rehabilitation',
            'home remedy', 'natural treatment', 'herbal', 'herbs',
            'supplement', 'supplements', 'vitamin', 'vitamins', 'mineral',
            'probiotic', 'probiotics', 'antioxidant', 'antioxidants',
            
            # Body Parts & Locations
            'head', 'forehead', 'temple', 'skull', 'brain', 'face',
            'eye', 'eyes', 'eyebrow', 'eyelid', 'eyelash', 'pupil',
            'ear', 'ears', 'earlobe', 'eardrum', 'hearing',
            'nose', 'nostril', 'nasal', 'sinus', 'sinuses',
            'mouth', 'lips', 'teeth', 'tooth', 'gums', 'tongue',
            'throat', 'pharynx', 'larynx', 'voice box', 'vocal cords',
            'neck', 'cervical', 'throat', 'windpipe', 'trachea',
            'chest', 'breast', 'breasts', 'rib', 'ribs', 'sternum',
            'back', 'spine', 'vertebrae', 'spinal cord', 'disc',
            'shoulder', 'shoulders', 'arm', 'arms', 'elbow', 'elbows',
            'wrist', 'wrists', 'hand', 'hands', 'finger', 'fingers',
            'thumb', 'thumb', 'nail', 'nails', 'palm', 'knuckle',
            'hip', 'hips', 'pelvis', 'pelvic', 'groin',
            'leg', 'legs', 'thigh', 'thighs', 'knee', 'knees',
            'ankle', 'ankles', 'foot', 'feet', 'toe', 'toes',
            'heel', 'heels', 'arch', 'sole', 'instep',
            'abdomen', 'stomach', 'belly', 'tummy', 'waist',
            'liver', 'kidney', 'kidneys', 'bladder', 'intestine',
            'lung', 'lungs', 'heart', 'brain', 'spine',
            
            # Emergency & Urgent Care
            'emergency', 'urgent', 'critical', 'acute', 'severe',
            'ambulance', 'paramedic', 'er', 'emergency room',
            'trauma', 'injury', 'injuries', 'wound', 'wounds',
            'cut', 'cuts', 'laceration', 'bruise', 'bruises', 'contusion',
            'burn', 'burns', 'scald', 'scalding', 'frostbite',
            'fracture', 'broken', 'break', 'sprain', 'strain',
            'dislocation', 'concussion', 'head injury', 'bleeding',
            'hemorrhage', 'shock', 'unconscious', 'unresponsive',
            'choking', 'drowning', 'poisoning', 'overdose',
            'allergic reaction', 'anaphylaxis', 'seizure', 'stroke',
            'heart attack', 'cardiac arrest', 'chest pain',
            'difficulty breathing', 'shortness of breath',
            
            # Women's Health
            'pregnancy', 'pregnant', 'prenatal', 'antenatal',
            'labor', 'delivery', 'birth', 'childbirth', 'miscarriage',
            'menstruation', 'menstrual', 'period', 'pms', 'menopause',
            'ovulation', 'fertility', 'infertility', 'contraception',
            'birth control', 'pills', 'iud', 'condom', 'condoms',
            'breastfeeding', 'lactation', 'mastitis', 'breast cancer',
            'cervical cancer', 'ovarian cancer', 'endometriosis',
            'fibroids', 'cysts', 'polycystic', 'pcos',
            
            # Men's Health
            'prostate', 'prostate cancer', 'testicular', 'testicular cancer',
            'erectile dysfunction', 'impotence', 'infertility',
            'male menopause', 'andropause', 'testosterone',
            
            # Children's Health
            'pediatric', 'pediatrics', 'child', 'children', 'baby', 'babies',
            'infant', 'infants', 'toddler', 'toddlers', 'adolescent',
            'teenager', 'teen', 'growth', 'development', 'milestone',
            'vaccination', 'immunization', 'shots', 'vaccines',
            'fever', 'teething', 'colic', 'diaper rash', 'cradle cap',
            'adhd', 'autism', 'developmental delay', 'learning disability',
            
            # Mental Health
            'mental health', 'psychiatric', 'psychological', 'psychology',
            'depression', 'depressed', 'sad', 'sadness', 'mood', 'moody',
            'anxiety', 'anxious', 'worry', 'worried', 'panic', 'panic attack',
            'phobia', 'phobias', 'fear', 'fears', 'trauma', 'ptsd',
            'bipolar', 'manic', 'mania', 'schizophrenia', 'psychosis',
            'eating disorder', 'anorexia', 'bulimia', 'binge eating',
            'addiction', 'alcoholism', 'drug addiction', 'substance abuse',
            'suicide', 'suicidal', 'self-harm', 'self injury',
            'therapy', 'counseling', 'psychotherapy', 'psychiatrist',
            'psychologist', 'therapist', 'counselor',
            
            # Nutrition & Diet
            'nutrition', 'nutritious', 'diet', 'dietary', 'food', 'eating',
            'calorie', 'calories', 'protein', 'carbohydrate', 'fat', 'fiber',
            'vitamin', 'vitamins', 'mineral', 'minerals', 'supplement',
            'supplements', 'herbal', 'herbs', 'natural', 'organic',
            'allergy', 'allergic', 'intolerance', 'lactose', 'gluten',
            'diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin',
            'obesity', 'overweight', 'underweight', 'weight loss',
            'weight gain', 'metabolism', 'metabolic', 'cholesterol',
            'sodium', 'salt', 'sugar', 'sweet', 'sweetener',
            'hydration', 'water', 'dehydration', 'dehydrated',
            'caffeine', 'alcohol', 'smoking', 'tobacco', 'nicotine',
            
            # Exercise & Fitness
            'exercise', 'fitness', 'workout', 'training', 'physical activity',
            'cardio', 'cardiovascular', 'strength', 'muscle', 'muscles',
            'flexibility', 'stretching', 'yoga', 'pilates', 'running',
            'walking', 'cycling', 'swimming', 'gym', 'gymnasium',
            'sports', 'athletic', 'athlete', 'performance', 'endurance',
            'rehabilitation', 'rehab', 'physical therapy', 'pt',
            'injury', 'injuries', 'prevention', 'warm-up', 'cool-down',
            
            # Aging & Elderly
            'aging', 'elderly', 'senior', 'seniors', 'geriatric', 'geriatrics',
            'dementia', 'alzheimer', 'memory', 'cognitive', 'cognition',
            'arthritis', 'osteoporosis', 'bone density', 'fracture',
            'fall', 'falls', 'balance', 'mobility', 'independence',
            'caregiver', 'caregiving', 'nursing home', 'assisted living',
            'medication', 'medications', 'polypharmacy', 'side effects',
            
            # Alternative Medicine
            'alternative medicine', 'complementary', 'holistic', 'natural',
            'homeopathy', 'acupuncture', 'chiropractic', 'massage',
            'meditation', 'mindfulness', 'relaxation', 'stress management',
            'aromatherapy', 'essential oils', 'herbal medicine', 'herbs',
            'supplements', 'vitamins', 'minerals', 'probiotics',
            'yoga', 'tai chi', 'qigong', 'reflexology', 'reiki',
            
            # Medical Equipment & Devices
            'stethoscope', 'thermometer', 'blood pressure cuff', 'monitor',
            'defibrillator', 'pacemaker', 'insulin pump', 'glucose meter',
            'inhaler', 'nebulizer', 'oxygen', 'ventilator', 'wheelchair',
            'walker', 'cane', 'crutches', 'brace', 'cast', 'splint',
            'bandage', 'bandages', 'gauze', 'tape', 'adhesive',
            'syringe', 'needle', 'needles', 'injection', 'iv', 'catheter',
            
            # Medical Specialties
            'cardiology', 'cardiac', 'dermatology', 'dermatologist',
            'endocrinology', 'endocrinologist', 'gastroenterology',
            'gastroenterologist', 'neurology', 'neurologist',
            'oncology', 'oncologist', 'orthopedics', 'orthopedist',
            'pediatrics', 'pediatrician', 'psychiatry', 'psychiatrist',
            'radiology', 'radiologist', 'surgery', 'surgeon',
            'urology', 'urologist', 'gynecology', 'gynecologist',
            'ophthalmology', 'ophthalmologist', 'otolaryngology',
            'ent', 'anesthesiology', 'anesthesiologist', 'pathology',
            'pathologist', 'emergency medicine', 'family medicine',
            'internal medicine', 'geriatrics', 'geriatrician'
        ]
        
        # Non-medical keywords
        self.non_medical_keywords = [
            # Mathematics & Science
            'mathematics', 'math', 'algebra', 'geometry', 'calculus', 'statistics',
            'physics', 'chemistry', 'biology', 'science', 'scientific', 'research',
            'experiment', 'laboratory', 'lab', 'hypothesis', 'theory', 'formula',
            'equation', 'calculation', 'computation', 'data', 'analysis',
            
            # History & Geography
            'history', 'historical', 'ancient', 'medieval', 'modern', 'war',
            'battle', 'empire', 'kingdom', 'civilization', 'culture', 'heritage',
            'geography', 'country', 'nation', 'continent', 'ocean', 'mountain',
            'river', 'city', 'capital', 'population', 'demographics',
            
            # Politics & Government
            'politics', 'political', 'government', 'election', 'vote', 'voting',
            'president', 'prime minister', 'parliament', 'congress', 'senate',
            'democracy', 'republic', 'monarchy', 'dictatorship', 'policy',
            'law', 'legal', 'court', 'judge', 'justice', 'rights', 'freedom',
            
            # Weather & Environment
            'weather', 'climate', 'temperature', 'rain', 'snow', 'storm',
            'hurricane', 'tornado', 'forecast', 'season', 'summer', 'winter',
            'spring', 'autumn', 'environment', 'pollution', 'global warming',
            'renewable energy', 'sustainability', 'conservation',
            
            # Sports & Recreation
            'sports', 'football', 'soccer', 'basketball', 'cricket', 'tennis',
            'golf', 'baseball', 'hockey', 'swimming', 'running', 'cycling',
            'gym', 'fitness', 'exercise', 'workout', 'training', 'competition',
            'tournament', 'championship', 'olympics', 'recreation', 'hobby',
            
            # Entertainment & Media
            'entertainment', 'movie', 'film', 'cinema', 'television', 'tv',
            'music', 'song', 'album', 'concert', 'dance', 'dancing',
            'art', 'painting', 'drawing', 'sculpture', 'gallery', 'museum',
            'theater', 'theatre', 'play', 'drama', 'comedy', 'actor',
            'actress', 'director', 'producer', 'celebrity', 'famous',
            
            # Food & Cooking
            'cooking', 'recipe', 'food recipe', 'restaurant', 'cuisine',
            'chef', 'kitchen', 'ingredients', 'meal', 'breakfast', 'lunch',
            'dinner', 'snack', 'beverage', 'drink', 'coffee', 'tea',
            'wine', 'beer', 'alcohol', 'diet', 'nutrition', 'calories',
            
            # Travel & Tourism
            'travel', 'tourism', 'vacation', 'holiday', 'trip', 'journey',
            'hotel', 'booking', 'reservation', 'flight', 'airplane', 'airport',
            'train', 'bus', 'car', 'driving', 'road', 'highway',
            'passport', 'visa', 'destination', 'sightseeing', 'tour',
            
            # Shopping & Commerce
            'shopping', 'buy', 'sell', 'purchase', 'price', 'cost', 'money',
            'store', 'shop', 'market', 'mall', 'online', 'ecommerce',
            'product', 'brand', 'advertisement', 'marketing', 'sales',
            'discount', 'offer', 'deal', 'bargain', 'payment', 'credit',
            
            # Technology & Computing
            'technology', 'computer', 'programming', 'coding', 'software',
            'hardware', 'internet', 'website', 'app', 'application',
            'mobile', 'phone', 'smartphone', 'tablet', 'laptop', 'desktop',
            'gaming', 'video game', 'console', 'ai', 'artificial intelligence',
            'machine learning', 'data science', 'cybersecurity', 'blockchain',
            
            # Finance & Economics
            'finance', 'banking', 'investment', 'stock', 'trading', 'market',
            'economy', 'economic', 'business', 'company', 'corporation',
            'profit', 'revenue', 'income', 'salary', 'wage', 'budget',
            'loan', 'credit', 'debt', 'insurance', 'tax', 'accounting',
            
            # Education & Learning
            'education', 'school', 'college', 'university', 'study', 'studying',
            'student', 'teacher', 'professor', 'academic', 'degree', 'diploma',
            'course', 'class', 'lesson', 'lecture', 'homework', 'exam',
            'test', 'grade', 'scholarship', 'tuition', 'campus',
            
            # Work & Career
            'job', 'career', 'employment', 'work', 'working', 'office',
            'meeting', 'project', 'task', 'deadline', 'boss', 'manager',
            'colleague', 'employee', 'employer', 'interview', 'resume',
            'application', 'hiring', 'firing', 'promotion', 'retirement',
            
            # Social & Relationships
            'relationship', 'dating', 'marriage', 'wedding', 'family',
            'friend', 'friendship', 'social', 'party', 'celebration',
            'birthday', 'anniversary', 'holiday', 'festival', 'event',
            'community', 'society', 'culture', 'tradition', 'custom',
            
            # Miscellaneous
            'fashion', 'clothing', 'style', 'beauty', 'cosmetics',
            'home', 'house', 'apartment', 'furniture', 'decoration',
            'garden', 'plant', 'flower', 'pet', 'animal', 'nature',
            'book', 'reading', 'writing', 'language', 'translation',
            'news', 'journalism', 'media', 'communication', 'conversation'
        ]
    
    def is_medical_query(self, query: str) -> bool:
        """Check if query is medical-related."""
        query_lower = query.lower()
        
        # Check for medical keywords
        has_medical_keywords = any(keyword in query_lower for keyword in self.medical_keywords)
        
        # Check for non-medical keywords
        has_non_medical_keywords = any(keyword in query_lower for keyword in self.non_medical_keywords)
        
        # If it has non-medical keywords and no medical keywords, it's not medical
        if has_non_medical_keywords and not has_medical_keywords:
            return False
        
        # If it has medical keywords, it's medical
        if has_medical_keywords:
            return True
        
        # Check for common medical question patterns
        medical_patterns = [
            'what is', 'how to', 'symptoms of', 'treatment for', 'cure for',
            'prevent', 'avoid', 'home remedy', 'natural treatment',
            'should i', 'can i', 'is it safe', 'side effects',
            'causes of', 'signs of', 'warning signs', 'when to see'
        ]
        
        has_medical_patterns = any(pattern in query_lower for pattern in medical_patterns)
        
        return has_medical_patterns
    
    def retrieve_medical_context(self, query: str) -> str:
        """Retrieve relevant medical context for RAG."""
        query_lower = query.lower()
        
        # Check for specific conditions
        for condition, info in self.medical_knowledge.items():
            if condition in query_lower:
                return info['info']
        
        # Check for related terms
        if any(term in query_lower for term in ['head', 'head pain', 'migraine']):
            return self.medical_knowledge['headache']['info']
        elif any(term in query_lower for term in ['stomach', 'belly', 'tummy', 'abdominal']):
            return self.medical_knowledge['stomach ache']['info']
        elif any(term in query_lower for term in ['knee', 'knees']):
            return self.medical_knowledge['knee pain']['info']
        elif any(term in query_lower for term in ['back', 'spine']):
            return self.medical_knowledge['back pain']['info']
        elif any(term in query_lower for term in ['period', 'menstrual', 'cramps']):
            return self.medical_knowledge['period pain']['info']
        elif any(term in query_lower for term in ['fever', 'temperature', 'hot']):
            return self.medical_knowledge['fever']['info']
        elif any(term in query_lower for term in ['cough', 'coughing']):
            return self.medical_knowledge['cough']['info']
        elif any(term in query_lower for term in ['throat', 'sore throat']):
            return self.medical_knowledge['sore throat']['info']
        elif any(term in query_lower for term in ['nausea', 'nauseous']):
            return self.medical_knowledge['nausea']['info']
        elif any(term in query_lower for term in ['tired', 'fatigue', 'exhausted']):
            return self.medical_knowledge['fatigue']['info']
        elif any(term in query_lower for term in ['stress', 'anxiety', 'worried']):
            return self.medical_knowledge['stress']['info']
        elif any(term in query_lower for term in ['sleep', 'insomnia', 'sleepless']):
            return self.medical_knowledge['insomnia']['info']
        elif any(term in query_lower for term in ['b12', 'b-12', 'vitamin b12']):
            return self.medical_knowledge['vitamin b12']['info']
        elif any(term in query_lower for term in ['water', 'thirst', 'dehydration']):
            return self.medical_knowledge['dehydration']['info']
        
        return "General medical information"
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using urllib with multiple model fallbacks."""
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        # Convert data to JSON
        json_data = json.dumps(data).encode('utf-8')
        
        # Try different API endpoints
        endpoints = [
            "https://generativelanguage.googleapis.com/v1beta/models",
            "https://generativelanguage.googleapis.com/v1/models"
        ]
        
        # Try each endpoint with each model
        for endpoint in endpoints:
            for model in self.available_models:
                try:
                    url = f"{endpoint}/{model}:generateContent"
                    
                    # Create request
                    req = urllib.request.Request(
                        f"{url}?key={self.api_key}",
                        data=json_data,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    # Make request
                    with urllib.request.urlopen(req) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        # Update the working model name
                        self.model_name = model
                        return result['candidates'][0]['content']['parts'][0]['text']
                        
                except urllib.error.HTTPError as e:
                    error_details = f"Model: {model}, Endpoint: {endpoint}, Code: {e.code}"
                    if e.code == 404:
                        # Try next model
                        continue
                    elif e.code == 403:
                        return f"API Error: Permission denied. {error_details}. Please check your API key permissions."
                    elif e.code == 400:
                        try:
                            error_body = e.read().decode('utf-8')
                            return f"API Error: Bad request. {error_details}. Response: {error_body}"
                        except:
                            return f"API Error: Bad request. {error_details}. Please check your API key and request format."
                    else:
                        return f"API Error: {e.code} - {error_details}. {str(e)}"
                except Exception as e:
                    # Try next model
                    continue
        
        # If all models and endpoints failed
        return f"API Error: All models and endpoints unavailable. Please check your API key and internet connection."
    
    def test_api_key(self) -> Dict[str, Any]:
        """Test if the API key is valid by making a simple request."""
        test_prompt = "Hello"
        response = self.call_gemini_api(test_prompt)
        
        if "API Error" in response or "Error calling Gemini API" in response:
            return {
                'valid': False,
                'error': response,
                'working_model': None
            }
        else:
            return {
                'valid': True,
                'error': None,
                'working_model': self.model_name
            }
    
    def test_simple_api_call(self) -> str:
        """Make a simple test call to verify API connectivity."""
        try:
            # Test with a very simple request
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            data = {
                "contents": [{
                    "parts": [{
                        "text": "Hi"
                    }]
                }]
            }
            
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                f"{url}?key={self.api_key}",
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return f"SUCCESS: {result['candidates'][0]['content']['parts'][0]['text']}"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function using RAG with Gemini API."""
        try:
            # Check if query is medical
            if not self.is_medical_query(query):
                return {
                    'response': "I'm a medical information assistant and can only help with health-related questions. Please ask me about medical topics, symptoms, treatments, or general health information.",
                    'is_medical': False,
                    'sources': [],
                    'disclaimer': self.disclaimer,
                    'model': self.model_name,
                    'rag': True
                }
            
            # Retrieve medical context (RAG)
            medical_context = self.retrieve_medical_context(query)
            
            # Create prompt for Gemini
            prompt = f"""You are a helpful medical information assistant. Provide general health information and home remedies based on the medical context below.

MEDICAL CONTEXT: {medical_context}

USER QUESTION: {query}

IMPORTANT RULES:
1. Only provide general health information and home remedies
2. Always include this disclaimer: "⚠️ IMPORTANT DISCLAIMER: I am not a medical professional. For diagnosis or treatment, consult a qualified healthcare provider."
3. Do not provide specific medical diagnoses or prescriptions
4. Encourage consulting healthcare professionals for serious conditions
5. Focus on general wellness and home care tips
6. Keep response concise but helpful

Please provide a helpful response with general health information and home remedies."""

            # Call Gemini API
            response_text = self.call_gemini_api(prompt)
            
            # Check if API call failed and provide fallback
            if "Error calling Gemini API" in response_text or "API Error" in response_text:
                # Fallback to RAG-only response
                fallback_response = f"""Based on medical knowledge, here's some general information:

{medical_context}

For your specific question: {query}

Please note that this is general health information only. For personalized medical advice, please consult with a qualified healthcare provider.

{self.disclaimer}"""
                
                return {
                    'response': fallback_response,
                    'is_medical': True,
                    'sources': ['rag_medical_knowledge_fallback'],
                    'disclaimer': self.disclaimer,
                    'model': 'RAG-only (API unavailable)',
                    'rag': True
                }
            
            # Ensure disclaimer is included
            if self.disclaimer not in response_text:
                response_text = f"{response_text}\n\n{self.disclaimer}"
            
            return {
                'response': response_text,
                'is_medical': True,
                'sources': ['rag_medical_knowledge'],
                'disclaimer': self.disclaimer,
                'model': self.model_name,
                'rag': True
            }
            
        except Exception as e:
            return {
                'response': f"I apologize, but I encountered an error while processing your request. Please try again or consult a healthcare professional.\n\n{self.disclaimer}",
                'is_medical': True,
                'sources': [],
                'disclaimer': self.disclaimer,
                'model': self.model_name,
                'rag': True,
                'error': str(e)
            }


def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FixedRAGGeminiMedicalChatbot()
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


def display_disclaimer():
    """Display the medical disclaimer banner."""
    st.markdown("""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336; margin-bottom: 20px;">
        <h4 style="color: #d32f2f; margin: 0;">⚠️ Important Medical Disclaimer</h4>
        <p style="margin: 10px 0 0 0; color: #424242;">
            <strong>I am not a medical professional.</strong> For diagnosis or treatment, consult a qualified healthcare provider. 
            This assistant provides general health information only and should not replace professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_conversation():
    """Display the conversation history."""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q: {exchange['query'][:50]}...", expanded=False):
                st.markdown(f"**Question:** {exchange['query']}")
                st.markdown(f"**Answer:** {exchange['response']}")
                if 'sources' in exchange and exchange['sources']:
                    st.markdown(f"**Sources:** {', '.join(exchange['sources'])}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏥 Medical AI Assistant")
    st.markdown("**Powered by Gemini 1.5 Flash with RAG** - Ask me about medical topics, symptoms, treatments, or general health information")
    
    # Display disclaimer
    display_disclaimer()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Model Selection
        st.subheader("🤖 AI Model Selection")
        selected_model = st.selectbox(
            "Choose Gemini Model:",
            options=st.session_state.chatbot.available_models,
            index=0,
            help="Select which Gemini model to use for responses"
        )
        
        # Update model if changed
        if selected_model != st.session_state.chatbot.model_name:
            st.session_state.chatbot.model_name = selected_model
            st.rerun()
        
        # Test API status
        api_status = st.session_state.chatbot.test_api_key()
        simple_test = st.session_state.chatbot.test_simple_api_call()
        
        if not api_status['valid']:
            st.warning("⚠️ API Unavailable - Using RAG Fallback")
            st.markdown("**Model:** RAG-only Mode")
            if api_status['error']:
                with st.expander("🔍 API Error Details", expanded=True):
                    st.code(api_status['error'])
            with st.expander("🔍 Simple API Test", expanded=False):
                st.code(simple_test)
        else:
            st.success("✅ Gemini API Ready")
            st.markdown(f"**Model:** {api_status['working_model']}")
            with st.expander("🔍 API Test Result", expanded=False):
                st.code(simple_test)
        
        st.markdown("**API:** Google Gemini")
        st.markdown("**RAG:** Retrieval-Augmented Generation")
        st.markdown("**Medical Knowledge:** Enhanced")
        st.markdown("**HTTP Library:** urllib (No conflicts)")
        
        # Model Information
        st.markdown("---")
        st.subheader("📋 Model Information")
        
        model_info = {
            "gemini-2.0-flash-exp": "Latest experimental model with enhanced capabilities",
            "gemini-1.5-pro": "High-performance model for complex tasks",
            "gemini-1.5-flash": "Fast and efficient model for quick responses",
            "gemini-pro": "Standard Gemini model with reliable performance",
            "gemini-1.5-flash-001": "Alternative flash model variant"
        }
        
        if selected_model in model_info:
            st.info(f"**{selected_model}**: {model_info[selected_model]}")
        else:
            st.info(f"**{selected_model}**: Custom model configuration")
        
        # Model switching note
        st.markdown("💡 **Note**: Changing models will test the new model automatically")
        
        # Model Status Check
        if st.button("🔄 Test All Models"):
            with st.spinner("Testing all available models..."):
                model_results = {}
                for model in st.session_state.chatbot.available_models:
                    # Temporarily set the model
                    original_model = st.session_state.chatbot.model_name
                    st.session_state.chatbot.model_name = model
                    
                    # Test the model
                    test_response = st.session_state.chatbot.call_gemini_api("Test")
                    model_results[model] = "✅ Working" if "API Error" not in test_response else "❌ Failed"
                    
                    # Restore original model
                    st.session_state.chatbot.model_name = original_model
                
                # Display results
                st.markdown("### 🧪 Model Test Results")
                for model, status in model_results.items():
                    st.markdown(f"**{model}**: {status}")
                
                # Auto-select first working model
                working_models = [model for model, status in model_results.items() if "✅" in status]
                if working_models and st.session_state.chatbot.model_name not in working_models:
                    st.session_state.chatbot.model_name = working_models[0]
                    st.success(f"🔄 Auto-selected working model: {working_models[0]}")
                    st.rerun()
        
        # API Key help section
        if not api_status['valid']:
            st.markdown("---")
            st.markdown("### 🔑 Get API Key")
            st.markdown("To enable Gemini AI features:")
            st.markdown("1. Go to [Google AI Studio](https://aistudio.google.com/)")
            st.markdown("2. Create a new project")
            st.markdown("3. Generate an API key")
            st.markdown("4. Set environment variable:")
            st.code("set GEMINI_API_KEY=your_api_key_here", language="bash")
            st.markdown("5. Restart the application")
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💬 Recent Questions")
            for exchange in st.session_state.conversation_history[-5:]:
                st.markdown(f"• {exchange['query'][:30]}...")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Medical Question")
        
        # Query input
        query = st.text_area(
            "Enter your health-related question:",
            placeholder="e.g., How to get rid of headache? What is vitamin B12? Home remedies for fever?",
            height=100,
            help="Ask about medical topics, symptoms, treatments, or general health information."
        )
        
        # Submit button
        if st.button("🔍 Get Medical Answer", type="primary"):
            if not query.strip():
                st.warning("Please enter a medical question.")
            else:
                with st.spinner("Generating response with Gemini AI and RAG..."):
                    # Get response from chatbot
                    response_data = st.session_state.chatbot.chat(query)
                    
                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'query': query,
                        'response': response_data['response'],
                        'sources': response_data.get('sources', []),
                        'is_medical': response_data.get('is_medical', False),
                        'model': response_data.get('model', 'unknown'),
                        'rag': response_data.get('rag', False)
                    })
                    
                    st.rerun()
    
    with col2:
        st.header("📊 Response Info")
        
        if st.session_state.conversation_history:
            latest = st.session_state.conversation_history[-1]
            
            # Display latest response
            st.markdown("### Latest Response")
            st.markdown(latest['response'])
            
            # Display metadata
            if 'sources' in latest and latest['sources']:
                st.markdown(f"**Sources:** {len(set(latest['sources']))} documents")
                st.markdown(f"**RAG:** {'Yes' if latest.get('rag') else 'No'}")
                st.markdown(f"**AI Model:** {latest.get('model', 'Unknown')}")
            
            # Show if it's medical or not
            if latest.get('is_medical'):
                st.success("✅ Medical Query Processed")
            else:
                st.warning("⚠️ Non-Medical Query Declined")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        display_conversation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>Medical AI Assistant | Powered by Gemini 1.5 Flash with RAG | For general health information only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
