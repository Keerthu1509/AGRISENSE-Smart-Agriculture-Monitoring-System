from google import genai
import os

client = genai.Client(api_key='AIzaSyCMY5-lrX7kPKBesjgQpm9O1bvO3jV65Io')

def get_treatment_recommendations(disease_name, crop_name, irrigation_status):
    
    prompt = f"""
You are an agricultural expert providing treatment recommendations for crop diseases.

Disease Detected: {disease_name}
Crop: {crop_name}
Irrigation Status: {irrigation_status}

Provide a comprehensive treatment plan in the following EXACT structured format:

Disease Analysis:
Provide 2-3 sentences about the disease, its causes, and impact on the crop.

Treatment Options:

Chemical Treatment:
A. [First chemical treatment recommendation in one line]
B. [Second chemical treatment recommendation in one line]
C. [Third chemical treatment recommendation in one line]

Organic Treatment:
A. [First organic treatment recommendation in one line]
B. [Second organic treatment recommendation in one line]
C. [Third organic treatment recommendation in one line]

Preventive Measures:
A. [First preventive measure in one line]
B. [Second preventive measure in one line]
C. [Third preventive measure in one line]
D. [Fourth preventive measure in one line]

Additional Recommendations:
A. [First additional recommendation in one line]
B. [Second additional recommendation in one line]
C. [Third additional recommendation considering irrigation status in one line]

CRITICAL: Do NOT use asterisks, hashtags, bullet points, or special formatting characters. Each point should be on a new line starting with A., B., C., etc. Keep each point concise (one line maximum).
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        recommendation = response.text
        
        recommendation = recommendation.replace('*', '')
        recommendation = recommendation.replace('#', '')
        recommendation = recommendation.replace('**', '')
        recommendation = recommendation.replace('###', '')
        
        return recommendation.strip()
    
    except Exception as e:
        return f"""Disease Analysis:
The detected disease is {disease_name} which commonly affects {crop_name} plants. This condition can significantly reduce crop yield if not treated promptly.

Treatment Options:

Chemical Treatment:
 Apply copper-based fungicides at recommended doses every 7-10 days
B. Spray chlorothalonil or mancozeb as per label instructions
C. Use systemic fungicides for severe infections

Organic Treatment:
A. Use neem oil spray diluted at 2-3 tablespoons per gallon of water
B. Apply Bacillus subtilis bio-fungicide weekly
C. Use sulfur-based organic fungicides as preventive measure

Preventive Measures:
A. Ensure proper plant spacing for air circulation
B. Remove and destroy infected plant debris immediately
C. Avoid overhead watering to reduce leaf wetness
D. Practice crop rotation with non-host plants

Additional Recommendations:
A. Monitor plants regularly for early disease symptoms
B. Maintain optimal soil nutrition to strengthen plant immunity
C. {irrigation_status} - adjust watering schedule to prevent favorable conditions for disease"""