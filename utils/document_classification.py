def match_words(word, cleaned_text):
    if word in cleaned_text:
        return True
    else:
        return False
        
def fuzzy_membership_score(cleaned_text, keyword_pairs):
    """
    Calculate a fuzzy membership score based on the presence of keywords.
    Keywords are provided as pairs (Nepali, English), and the score is computed based on the presence of either keyword.
    """
    score = sum(1 for nepali, english in keyword_pairs if (match_words(nepali,cleaned_text)) or (match_words(english,cleaned_text)))
    return score / len(keyword_pairs) if keyword_pairs else 0.0

def classify_document_fuzzy(cleaned_text):
    # Classify the document using fuzzy logic based on keyword presence.#
    # Define keyword sets for different document types
    # List of keywords and phrases commonly found in a Nepali Citizenship Certificate
    citizenship_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("जिल्ला प्रशासन कार्यालय","District Administration Office"),
        ("गृह मन्त्रालय", "Ministry of Home Affairs"),
        ("जन्म मिति", "Date of Birth"),
        ("नागरिकताको प्रकार", "Type of Citizenship"),
        ("स्थायी बासस्थान", "Permanent Address"),
        ("जन्मस्थान","Birth Place"),
        ("नाम", "Name"),
        ("थर","Surname"),
        ("बाबुको नाम", "Father's Name"),
        ("आमाको नाम", "Mother's Name"),
        ("पतिको नाम","Husband's Name"),
        ("पत्नीको नाम","Wife's Name"),
        ("लिङ्ग", "Gender"),
        ("नेपाली नागरिकताको प्रमाणपत्र", "Nepali Citizenship Certificate"),
        ("ठेगाना", "Address"),
        ("जिल्ला", "District"),
        ("नागरिकता","Citizenship")
    ]

    # List of keywords and phrases commonly found in a PAN Card
    pan_card_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("आन्तरिक राजस्व विभाग", "Inland Revenue Department"),
        ("Permanent Account Number", "Permanent Account Number"),
        ("नाम", "Name"),
        ("जन्म मिति", "Date of Birth"),
        ("स्थायी.ले.न.", "PAN"),
        ("ठेगाना", "Address"),
        ("मिति", "Date"),
        ("सही", "Signature")
    ]

    # List of keywords and phrases commonly found in a Passport
    passport_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("विदेश मन्त्रालय", "Ministry of Foreign Affairs"),
        ("Passport Number", "Passport Number"),
        ("नाम", "Name"),
        ("लिङ्ग", "Gender"),
        ("राष्ट्रियता", "Nationality"),
        ("जन्म मिति", "Date of Birth"),
        ("जन्मस्थान", "Place of Birth"),
        ("पेसा", "Profession"),
        ("स्थायी ठेगाना", "Permanent Address"),
        ("पासपोर्ट जारी मिति", "Passport Issue Date"),
        ("पासपोर्ट समाप्त मिति", "Passport Expiry Date"),
        ("जारी गर्ने प्राधिकरण", "Issuing Authority"),
        ("फोटो", "Photograph"),
        ("हस्ताक्षर", "Signature")
    ]
    # List of keywords and phrases commonly found in a Driving License
    driving_license_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("सवारी चालक अनुमतिपत्र", "Driving License"),
        ("नाम", "Name"),
        ("ठेगाना", "Address"),
        ("लाइसेन्स कार्यालय", "License Office"),
        ("जन्म मिति", "DOB"),
        ("नागरिकता नम्बर", "Citizenship No"),
        ("पासपोर्ट नम्बर", "Passport No"),
        ("फोन नम्बर", "Phone No"),
        ("जारी गर्ने", "Issued by"),
        ("धारकको हस्ताक्षर", "Signature of Holder")
    ]


    # Calculate fuzzy membership scores
    citizenship_score = fuzzy_membership_score(cleaned_text, citizenship_keywords)
    pan_card_score = fuzzy_membership_score(cleaned_text, pan_card_keywords)
    passport_score = fuzzy_membership_score(cleaned_text, passport_keywords)
    driving_license_score = fuzzy_membership_score(cleaned_text,driving_license_keywords)

    # Prepare a dictionary to store scores and corresponding classifications
    scores = {
        "Nepali Citizenship Document": citizenship_score,
        "PAN Card": pan_card_score,
        "Passport": passport_score,
        "Driving License": driving_license_score
    }

    # Determine the classification based on the highest score
    classification = "Unknown Document"
    confidence = 0.0
    # Get the highest score classification
    highest_classification = max(scores, key=scores.get)
    print(scores[highest_classification],highest_classification)

    # Check if the highest score is above the threshold
    if scores[highest_classification] > 0.4:
        classification = f"{highest_classification} Detected"
        confidence = scores[highest_classification]

    return classification, confidence
