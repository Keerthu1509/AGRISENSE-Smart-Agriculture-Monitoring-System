from twilio.rest import Client

# Twilio Credentials - REPLACE WITH YOUR ACTUAL CREDENTIALS
# You mentioned Organization SID: OR95f7baa2d846afe870062ca686a96168
# However, the Python client requires Account SID (AC...) and Auth Token.
TWILIO_ACCOUNT_SID = 'AC64a01f0b4386dee924caa31d036736a7' 
TWILIO_AUTH_TOKEN = '42979a94266c59d23ab63512e88b99d1'
TWILIO_PHONE_NUMBER = '+14179483291' 

def send_sms(to_number, message_body):
    """
    Sends an SMS message using Twilio.
    """
    try:
        # Check if credentials are still placeholders
        if TWILIO_ACCOUNT_SID == 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX':
            print("Twilio credentials are placeholders. SMS not sent.")
            print(f"Would have sent to {to_number}: {message_body}")
            # Return a fake SID for testing UI flow if desired, or None
            # return "SM_MOCK_SID_12345" 
            return None
            
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Add country code if missing (assuming India +91 base on user context)
        if not to_number.startswith('+'):
            to_number = f"+91{to_number}"

        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        
        print(f"SMS sent successfully! SID: {message.sid}")
        return message.sid
        
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return None
