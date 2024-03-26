import PIL.Image as Image
import torch
from model_architecture import Food101_V0, Food101_V1, Food101_V2, Food101_V3
from torchvision import transforms

def preprocess_image(image):
    img_size = 224
    data_transform = transforms.Compose([
        # Resize the images into 64x64
        transforms.Resize(size=(img_size, img_size)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transform(image)

def label_to_food(label)->str:
    label_dict = {'apple_pie': 0,
        'baby_back_ribs': 1,
        'baklava': 2,
        'beef_carpaccio': 3,
        'beef_tartare': 4,
        'beet_salad': 5,
        'beignets': 6,
        'bibimbap': 7,
        'bread_pudding': 8,
        'breakfast_burrito': 9,
        'bruschetta': 10,
        'caesar_salad': 11,
        'cannoli': 12,
        'caprese_salad': 13,
        'carrot_cake': 14,
        'ceviche': 15,
        'cheese_plate': 16,
        'cheesecake': 17,
        'chicken_curry': 18,
        'chicken_quesadilla': 19,
        'chicken_wings': 20,
        'chocolate_cake': 21,
        'chocolate_mousse': 22,
        'churros': 23,
        'clam_chowder': 24,
        'club_sandwich': 25,
        'crab_cakes': 26,
        'creme_brulee': 27,
        'croque_madame': 28,
        'cup_cakes': 29,
        'deviled_eggs': 30,
        'donuts': 31,
        'dumplings': 32,
        'edamame': 33,
        'eggs_benedict': 34,
        'escargots': 35,
        'falafel': 36,
        'filet_mignon': 37,
        'fish_and_chips': 38,
        'foie_gras': 39,
        'french_fries': 40,
        'french_onion_soup': 41,
        'french_toast': 42,
        'fried_calamari': 43,
        'fried_rice': 44,
        'frozen_yogurt': 45,
        'garlic_bread': 46,
        'gnocchi': 47,
        'greek_salad': 48,
        'grilled_cheese_sandwich': 49,
        'grilled_salmon': 50,
        'guacamole': 51,
        'gyoza': 52,
        'hamburger': 53,
        'hot_and_sour_soup': 54,
        'hot_dog': 55,
        'huevos_rancheros': 56,
        'hummus': 57,
        'ice_cream': 58,
        'lasagna': 59,
        'lobster_bisque': 60,
        'lobster_roll_sandwich': 61,
        'macaroni_and_cheese': 62,
        'macarons': 63,
        'miso_soup': 64,
        'mussels': 65,
        'nachos': 66,
        'omelette': 67,
        'onion_rings': 68,
        'oysters': 69,
        'pad_thai': 70,
        'paella': 71,
        'pancakes': 72,
        'panna_cotta': 73,
        'peking_duck': 74,
        'pho': 75,
        'pizza': 76,
        'pork_chop': 77,
        'poutine': 78,
        'prime_rib': 79,
        'pulled_pork_sandwich': 80,
        'ramen': 81,
        'ravioli': 82,
        'red_velvet_cake': 83,
        'risotto': 84,
        'samosa': 85,
        'sashimi': 86,
        'scallops': 87,
        'seaweed_salad': 88,
        'shrimp_and_grits': 89,
        'spaghetti_bolognese': 90,
        'spaghetti_carbonara': 91,
        'spring_rolls': 92,
        'steak': 93,
        'strawberry_shortcake': 94,
        'sushi': 95,
        'tacos': 96,
        'takoyaki': 97,
        'tiramisu': 98,
        'tuna_tartare': 99,
        'waffles': 100}
    label_list = list(label_dict.keys())
    return label_list[label]

def get_confidence(logits: torch.Tensor) -> float:
    softmax = torch.nn.Softmax(dim =1)
    confidence = torch.max(softmax(logits))
    return confidence*100

def predict_V0(processed_image):
    # device = "cpu"
    model = Food101_V0(3*224*224, 101, 1024)
    model.load_state_dict(torch.load('models/model_V0.pth'), map_location =torch.load('cpu'))
    model.eval()

    with torch.inference_mode():
        logits = model(torch.unsqueeze(processed_image, 0))
        confidence = get_confidence(logits)
        label = logits.argmax(dim=1)
        # print(label)
        return label[0], confidence
    

def predict_V1(processed_image):
    device = "cpu"
    model = Food101_V1(3, 32, 101).to(device)
    model.load_state_dict(torch.load('models/model_V1.pth'))
    model.eval()

    with torch.inference_mode():
        logits = model(torch.unsqueeze(processed_image, 0).to(device))
        confidence = get_confidence(logits)
        label = logits.argmax(dim=1)
        # print(label)
        return label[0], confidence
    
def predict_V2(processed_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Food101_V2.to(device)
    model.load_state_dict(torch.load('models/model_V2.pth'))
    model.eval()

    with torch.inference_mode():
        logits = model(torch.unsqueeze(processed_image, 0).to(device))
        confidence = get_confidence(logits)
        label = logits.argmax(dim=1)
        # print(label)
        return label[0], confidence

def predict_V3(processed_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Food101_V3.to(device)
    model.load_state_dict(torch.load('models/model_V3.pth'))
    model.eval()

    with torch.inference_mode():
        logits = model(torch.unsqueeze(processed_image, 0).to(device))
        confidence = get_confidence(logits)
        label = logits.argmax(dim=1)
        # print(label)
        return label[0], confidence
    
if __name__ == '__main__':
    image_path = 'data/food-101/food-101/images/apple_pie/134.jpg'
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)
    label, confidence = predict_V2(preprocessed_image)
    print(f'Predicted label : {label} with confidence {confidence: .4f}%')
    print(f'Corresponding food : {label_to_food(label)}')