def match_words(word, cleaned_text):
    # for text in cleaned_text:
    #     matched_characters = convert_words_to_char_sets(word).intersection(convert_words_to_char_sets(text))
    #     similarity_ratio = len(matched_characters) / min(len(word), len(text))
    #     if similarity_ratio > 0.7:
    #         break

    # if similarity_ratio > 0.7:
    #     return true
    # else:
    #     return false
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
        ("गृह मन्त्रालय", "Ministry of Home Affairs"),
        ("जन्म मिति", "Date of Birth"),
        ("नागरिकताको प्रकार", "Type of Citizenship"),
        ("स्थायी ठेगाना", "Permanent Address"),
        ("नाम", "Name"),
        ("बाबुको नाम", "Father's Name"),
        ("आमाको नाम", "Mother's Name"),
        ("लिङ्ग", "Gender"),
        ("नेपाली नागरिकता प्रमाणपत्र", "Nepali Citizenship Certificate"),
        ("ठेगाना", "Address"),
        ("जिल्ला", "District")
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
        ("जन्म मिति", "Date of Birth"),
        ("नागरिकता नम्बर", "Citizenship No."),
        ("पासपोर्ट नम्बर", "Passport No."),
        ("फोन नम्बर", "Phone No."),
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

    # Check if the highest score is above the threshold
    if scores[highest_classification] > 0.4:
        classification = f"{highest_classification} Detected"
        confidence = scores[highest_classification]

    return classification, confidence

#main function controlling overall workflow000000
def main():
    try:
        # Set up tkinter root
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open a file dialog to select an image
        image_path = filedialog.askopenfilename(title="Select an Image", 
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_path:  # Check if the user canceled the dialog
            print("No image selected.")
            return
        
        # Load and display the original document
        image = load_image(image_path)
        cv2.imshow('Original Document', image)
        
        cv2.waitKey(0)

        # Pre-process the image
        gray, mask = preprocess_image(image)
        cv2.imshow('Grayscaled Document', gray)
        cv2.waitKey(0)
        cv2.imshow('Threshold Binary Document', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract text from the mask
        text = extract_text(mask)\

        # Classify the document using fuzzy logic and print results
        classification_result, confidence = classify_document_fuzzy(text)
        print(f"Extracted Text:\n{text}")
        print(f"Classification Result: {classification_result}")
        print(confidence)
        confidence_level = confidence * 100
        print(f"Confidence Level in %: {confidence_level:.2f}")

    except Exception as e:
        print(e)
