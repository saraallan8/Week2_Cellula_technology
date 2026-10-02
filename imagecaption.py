from transformers import BlipProcessor, BlipForConditionalGeneration

#prepare the uploaded imgage so BLIB can undestand it
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def generate_caption(image):
    inputs = processor(images=image,
                       return_tensors="pt")

    output = model.generate( **inputs,
                            max_new_tokens = 50)

    caption = processor.decode(output[0],
                               skip_special_token = True)

    return caption