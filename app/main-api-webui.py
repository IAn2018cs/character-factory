import argparse
import base64
import os
from io import BytesIO
from typing import Optional

import aichar
import gradio as gr
import requests
from PIL import Image

supported_ollama_models = [
    "llama3:latest",
    "mistral:latest",
    "phi3:medium",
    "qwen:7b",
    "gemma:7b",
    "zephyr:7b",
]

supported_openai_models = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]


def generate_by_ollama(model: str, messages: list[dict]) -> Optional[str]:
    try:
        url = f'{args.ollama_host}/api/chat'
        data = {
            "model": model,
            "messages": messages,
            "options": {
                "repeat_penalty": 1.1,
                "top_k": 40,
                "top_p": 0.95,
                "temperature": 0.8
            },
            "stream": False
        }
        response = requests.post(url, json=data)
        return response.json()['message']['content']
    except Exception as e:
        print(f"generate_by_ollama error: {e}")
        return None


def generate_by_openai(model: str, messages: list[dict]) -> Optional[str]:
    try:
        url = f'{args.openai_host}/v1/chat/completions'
        data = {
            "model": model,
            "messages": messages,
            "options": {
                "frequency_penalty": 1.1,
                "temperature": 0.8
            },
            "stream": False
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {args.openai_key}"
            },
            json=data
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"generate_by_openai error: {e}")
        return None


def generate_image_by_sd(model: str, prompt: str, negative_prompt: str,
                         steps: int, cfg: float,
                         sampler_name: str, scheduler: str,
                         width: int, height: int) -> Optional[Image]:
    try:
        url = f'{args.sd_host}/sdapi/v1/txt2img'
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg,
            "width": width,
            "height": height,
            "batch_size": 1,
            "n_iter": 1,
            "seed": -1,
            "override_settings": {
                "sd_model_checkpoint": model
            },
            "save_images": False,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        }
        response = requests.post(url, json=data)
        print(response.json()['info'])
        images = response.json()['images']
        im = Image.open(BytesIO(base64.b64decode(images[0])))
        return im
    except Exception as e:
        print(f"generate_by_openai error: {e}")
        return None


def generate_character_name(model: str, topic: str, gender: str):
    gender = input_none(gender)
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool, you should always just return the name of the character and nothing else, you should not ask any questions.\n"
                       + "You only answer by giving the name of the character, you do not describe it, you do not mention anything about it. You can't write anything other than the character's name."
        },
        {
            "role": "user",
            "content": "Generate a random character name. Topic: business. Gender: male"
        },
        {
            "role": "assistant",
            "content": "Jamie Hale"
        },
        {
            "role": "user",
            "content": "Generate a random character name. Topic: fantasy"
        },
        {
            "role": "assistant",
            "content": "Eldric"
        },
        {
            "role": "user",
            "content": "Generate a random character name. Topic: anime. Gender: female"
        },
        {
            "role": "assistant",
            "content": "Tatsukaga Yamari"
        },
        {
            "role": "user",
            "content": "Generate a random character name. Topic: {{user}}'s pet cat."
        },
        {
            "role": "assistant",
            "content": "mr. Fluffy"
        },
        {
            "role": "user",
            "content": f"Generate a random character name. Topic: {topic}. {'Character gender: ' + gender + '.' if gender else ''} "
        }
    ]
    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"character_name: {output}")
    return output


def generate_character_summary(model: str, character_name: str, topic: str, gender: str):
    gender = input_none(gender)
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool. Describe the character in a very simple and understandable way, you can just list some characteristics, you do not need to write a professional characterization of the character. Describe: age, height, personality traits, appearance, clothing, what the character likes, what the character does not like.\n"
                       + "You must not write any summaries, overalls, endings or character evaluations at the end, you just have to return the character's personality and physical traits.\n"
                       + "Don't ask any questions, don't inquire about anything.\n"
                       + "The topic given by the user is to serve as a background to the character, not as the main theme of your answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style. If user gives as the topic eg. 'noir style detective', you do not return things like:\n"
                       + "'Character is a noir style detective', you just describe it so that the character fits that theme. Use simple and understandable English, use simple and colloquial terms.\n"
                       + "You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.\n"
                       + "Should include in its description who the character is - for example, a human mage, an elf archer, a shiba dog.\n"
                       + "Should be in the same form as the previous answers.\n"
                       + "You must include character traits, physical and character. You can't add anything else."
        },
        {
            "role": "user",
            "content": "Create a shorter description for a character named Tatsukaga Yamari. Character gender: female. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of anime but don't specify what topic it is, and don't describe the topic itself. You are to write a brief\n"
                       + "description of the character, do not write any summaries."
        },
        {
            "role": "assistant",
            "content": "Tatsukaga Yamari is a anime girl, she is 23 year old, is a friendly and cheerful person, is always helpful, Has a nice and friendly relationship with other people.\n"
                       + "She is tall and has long red hair. Wears an anime schoolgirl outfit in blue colors. She likes to read books in solitude, or in the presence of a maximum of a few people, enjoys coffee lattes, and loves cats and kitties. She does not like stressful situations, bitter coffee, dogs.\n"
                       + "Tatsukaga Yamari loves: being helpful, being empathetic, making new friends, spend time in silence reading science books, loves latte coffee\n"
                       + "Tatsukaga Yamari hates: apathy towards people, coffee without sugar and milk, espresso, noisy parties, disagreements between people, dogs, being alone\n"
                       + "Tatsukaga Yamari abilities: Smarter than her peers, keeping calm for a long time, quickly forgiving other people\n"
        },
        {
            "role": "user",
            "content": "Create a shorter description for a character named mr. Fluffy. Describe their appearance, distinctive features, and looks. Tailor the character to the theme of {{user}}'s pet cat but don't specify what topic it is, and don't describe the topic itself. You are to write a brief description of the\n"
                       + "character, do not write any summaries."
        },
        {
            "role": "assistant",
            "content": "Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy\n"
                       + "and expresses himself in a very sarcastic way.\n"
                       + "Mr Fluffy loves: good food, Being more intelligent and smarter than other people, learning philosophy and abstract concepts, spending time with {{user}}, he likes to lie lazily on his side\n"
                       + "Mr Fluffy hates: cheap food, loud people\n"
                       + "Mr Fluffy abilities: An ordinary domestic cat with the ability to speak and incredible knowledge of philosophy, Can eat incredible amounts of (good) food and not feel satiated"
        },
        {
            "role": "user",
            "content": f"Create a longer description for a character named {character_name}. {'Character gender: ' + gender + '.' if gender else ''} "
                       + "Describe their appearance, distinctive features, and looks. "
                       + f"Tailor the character to the theme of {topic} but don't "
                       + "specify what topic it is, and don't describe the topic itself. "
                       + "You are to write a brief description of the character. You must "
                       + "include character traits, physical and character. You can't add "
                       + "anything else. You must not write any summaries, conclusions or endings."
        }
    ]
    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"character_summary: {output}")
    return output


def generate_character_personality(model: str, character_name: str, character_summary: str, topic: str):
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool. Describe the character personality in a very simple and understandable way.\n"
                       + "You can simply list the most suitable character traits for a given character, the user-designated character description as well as the theme can help you in matching personality traits.\n"
                       + "Don't ask any questions, don't inquire about anything.\n"
                       + "You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.\n"
                       + "Don't write any summaries, endings or character evaluations at the end, you just have to return the character's personality traits. Use simple and understandable English, use simple and colloquial terms.\n"
                       + "You are not supposed to write characterization of the character, you don't have to form terms whether the character is good or bad, only you are supposed to write out the character traits of that character, nothing more.\n"
                       + "You must return character traits in your answers, you can not describe the appearance, clothing, or who the character is, only character traits.\n"
                       + "Your answer should be in the same form as the previous answers."
        },
        {
            "role": "user",
            "content": "Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him"
        },
        {
            "role": "assistant",
            "content": "Jamie Hale is calm, stoic, focused, intelligent, sensitive to art, discerning, focused, motivated, knowledgeable about business, knowledgeable about new business technologies, enjoys reading business and science books"
        },
        {
            "role": "user",
            "content": "Describe the personality of Mr Fluffy. Their characteristics  Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy and expresses himself in a very sarcastic way"
        },
        {
            "role": "assistant",
            "content": "Mr Fluffy is small, calm, lazy, mischievous cat, speaks in a very philosophical manner and is very sarcastic in his statements, very intelligent for a cat and even for a human, has a vast amount of knowledge about philosophy and the world"
        },
        {
            "role": "user",
            "content": f"Describe the personality of {character_name}. "
                       + f"Their characteristic {character_summary}\nDescribe them "
                       + "in a way that allows the reader to better understand their "
                       + "character. Make this character unique and tailor them to "
                       + f"the theme of {topic} but don't specify what topic it is, "
                       + "and don't describe the topic itself. You are to write out "
                       + "character traits separated by commas, you must not write "
                       + "any summaries, conclusions or endings."
        }
    ]

    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"character_personality: {output}")
    return output


def generate_character_scenario(model: str, character_summary: str, character_personality: str, topic: str):
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool.\n"
                       + "The topic given by the user is to serve as a background to the character, not as the main theme of your answer.\n"
                       + "Use simple and understandable English, use simple and colloquial terms.\n"
                       + "You must include {{user}} and {{char}} in your response.\n"
                       + "Your answer must be very simple and tailored to the character, character traits and theme.\n"
                       + "Your answer must not contain any dialogues.\n"
                       + "Instead of using the character's name you must use {{char}}.\n"
                       + "Your answer should be in the same form as the previous answers.\n"
                       + "Your answer must be short, maximum 5 sentences.\n"
                       + "You can not describe the character, but you have to describe the scenario and actions."
        },
        {
            "role": "user",
            "content": "Write a simple and undemanding introduction to the story, in which the main characters will be {{user}} and {{char}}, do not develop the story, write only the introduction. {{char}} characteristics: Tatsukaga Yamari is an 23 year old anime girl, who loves books and coffee. Make this character unique and tailor them to the theme of anime, but don't specify what topic it is, and don't describe the topic itself. Your response must end when {{user}} and {{char}} interact."
        },
        {
            "role": "assistant",
            "content": "When {{user}} found a magic stone in the forest, he moved to the magical world, where he meets {{char}}, who looks at him in disbelief, but after a while comes over to greet him."
        },
        {
            "role": "user",
            "content": "Write a scenario for chat roleplay "
                       + "to serve as a simple storyline to start chat "
                       + "roleplay by {{char}} and {{user}}. {{char}} "
                       + f"characteristics: {character_summary}. "
                       + f"{character_personality}. Make this character unique "
                       + f"and tailor them to the theme of {topic} but don't "
                       + "specify what topic it is, and don't describe the topic "
                       + "itself. Your answer must not contain any dialogues. "
                       + "Your response must end when {{user}} and {{char}} interact."
        }
    ]

    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"character_scenario: {output}")
    return output


def generate_character_greeting_message(
        model: str, character_name: str, character_summary: str,
        character_personality: str, topic: str
):
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool, you are supposed to generate answers so that they are simple and clear. You play the provided character and you write a message that you would start a chat roleplay with {{user}}. The form of your answer should be similar to previous answers.\n"
                       + "The topic given by the user is only to be an aid in selecting the style of the answer, not the main purpose of the answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style.\n"
                       + "You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on."
        },
        {
            "role": "user",
            "content": "Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and \n"
                       + "caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself"
        },
        {
            "role": "assistant",
            "content": "*Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{user}}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today?"
        },
        {
            "role": "user",
            "content": "Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.\n"
                       + "Jamie's appearance is always polished and professional.\n"
                       + "Jamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd \n"
                       + "decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\n"
                       + "greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself"
        },
        {
            "role": "assistant",
            "content": "*Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{user}}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success?"
        },
        {
            "role": "user",
            "content": "Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\n"
                       + "Eldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat\n"
                       + "abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\n"
                       + "greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of fantasy but don't specify what topic it is, and don't describe the topic itself"
        },
        {
            "role": "assistant",
            "content": "*Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{user}}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today?"
        },
        {
            "role": "user",
            "content": "Create the first message that the character "
                       + f"{character_name}, whose personality is "
                       + f"{character_summary}\n{character_personality}\n "
                       + "greets the user we are addressing as {{user}}. "
                       + "Make this character unique and tailor them to the theme "
                       + f"of {topic} but don't specify what topic it is, "
                       + "and don't describe the topic itself. You must match the "
                       + "speaking style to the character, if the character is "
                       + "childish then speak in a childish way, if the character "
                       + "is serious, philosophical then speak in a serious and "
                       + "philosophical way, and so on."
        }
    ]

    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"character_greeting_message: {output}")
    return output


def generate_example_messages(
        model: str,
        character_name: str, character_summary: str, character_personality: str, topic: str
):
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool, you are supposed to generate answers so that they are simple and clear.\n"
                       + "Your answer should be a dialog between {{user}} and {{char}}, where {{char}} is the specified character. The dialogue must be several messages taken from the roleplay chat between the user and the character.\n"
                       + "Only respond in {{user}} or {{char}} messages. The form of your answer should be similar to previous answers.\n"
                       + "You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.\n"
                       + "If the character is shy, then needs to speak little and quietly, if the character is aggressive then needs to shout and speak a lot and aggressively, if the character is sad then needs to be thoughtful and quiet, and so on.\n"
                       + "Dialog of {{user}} and {{char}} must be appropriate to their character traits and the way they speak.\n"
                       + "Instead of the character's name you must use {{char}}."
        },
        {
            "role": "user",
            "content": "Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself"
        },
        {
            "role": "assistant",
            "content": "{{user}}: Good afternoon, Mr. {{char}}. I've heard so much about your success in the corporate world. It's an honor to meet you.\n"
                       + "{{char}}: *{{char}} gives a warm smile and extends his hand for a handshake.* The pleasure is mine, {{user}}. Your reputation precedes you. Let's make this venture a success together.\n"
                       + "{{user}}: *Shakes {{char}}'s hand with a firm grip.* I look forward to it.\n"
                       + "{{char}}: *As they release the handshake, Jamie leans in, his eyes sharp with interest.* Impressive. Tell me more about your innovations and how they align with our goals."
        },
        {
            "role": "user",
            "content": "Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Tatsukaga Yamari. Tatsukaga Yamari characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself"
        },
        {
            "role": "assistant",
            "content": "{{user}}: {{char}}, this forest is absolutely enchanting. What's the plan for our adventure today?\n"
                       + "{{char}}: *{{char}} grabs {{user}}'s hand and playfully twirls them around before letting go.* Well, we're off to the Crystal Caves to retrieve the lost Amethyst Shard. It's a treacherous journey, but I believe in us.\n"
                       + "{{user}}: *Nods with determination.* I have no doubt we can do it. With your magic and our unwavering friendship, there's nothing we can't accomplish.\n"
                       + "{{char}}: *{{char}} moves closer, her eyes shining with trust and camaraderie.* That's the spirit, {{user}}! Let's embark on this epic quest and make the Crystal Caves ours!"
        },
        {
            "role": "user",
            "content": "Create a dialogue between {{user}} and {{char}}, "
                       + "they should have an interesting and engaging conversation, "
                       + "with some element of interaction like a handshake, movement, "
                       + "or playful gesture. Make it sound natural and dynamic. "
                       + "{{char}} is "
                       + f"{character_name}. {character_name} characteristics: "
                       + f"{character_summary}. {character_personality}. Make this "
                       + f"character unique and tailor them to the theme of {topic} but "
                       + "don't specify what topic it is, and don't describe the "
                       + "topic itself. You must match the speaking style to the character, "
                       + "if the character is childish then speak in a childish way, if the "
                       + "character is serious, philosophical then speak in a serious and "
                       + "philosophical way and so on."
        }
    ]

    if model in supported_ollama_models:
        output = generate_by_ollama(model, messages)
    else:
        output = generate_by_openai(model, messages)
    print(f"example_messages: {output}")
    return output


def generate_character_avatar(
        model: str,
        character_name: str,
        character_summary: str,
        topic: str,
        negative_prompt: str,
        avatar_prompt: str
):
    messages = [
        {
            "role": "system",
            "content": "You are a text generation tool, in the response you are supposed to give only descriptions of the appearance, what the character looks like, describe the character simply and unambiguously"
        },
        {
            "role": "user",
            "content": "create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.\n"
                       + "Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business"
        },
        {
            "role": "assistant",
            "content": "male, realistic, human, Confident and commanding presence, Polished and professional appearance, tailored suit, Well-maintained physique, Dark well-groomed hair, Neatly trimmed beard, blue eyes"
        },
        {
            "role": "user",
            "content": "create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.\n"
                       + "Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows,\n"
                       + "cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime"
        },
        {
            "role": "assistant",
            "content": "female, anime, Petite and delicate frame, Raven-black hair flowing down to her waist, Striking purple ribbon in her hair, Large and expressive amethyst-colored eyes, Colorful and eclectic outfit, oversized bows, cat-eared headbands, mismatched socks"
        },
        {
            "role": "user",
            "content": "create a prompt that lists the appearance "
                       + "characteristics of a character whose summary is "
                       + f"{character_summary}. Topic: {topic}"
        }
    ]
    sd_prompt = (
            input_none(avatar_prompt)
            or (
                generate_by_ollama(model, messages) if model in supported_ollama_models
                else generate_by_openai(model, messages)
            )
    )
    print(f"character_avatar sd_prompt: {sd_prompt}")
    return image_generate(character_name, sd_prompt, input_none(negative_prompt))


def image_generate(character_name: str, prompt: str, negative_prompt: str):
    prompt = "absurdres, full hd, 8k, high quality, " + prompt
    default_negative_prompt = (
            "worst quality, normal quality, low quality, low res, blurry, "
            + "text, watermark, logo, banner, extra digits, cropped, "
            + "jpeg artifacts, signature, username, error, sketch, "
            + "duplicate, ugly, monochrome, horror, geometry, "
            + "mutation, disgusting, "
            + "bad anatomy, bad hands, three hands, three legs, "
            + "bad arms, missing legs, missing arms, poorly drawn face, "
            + " bad face, fused face, cloned face, worst face, "
            + "three crus, extra crus, fused crus, worst feet, "
            + "three feet, fused feet, fused thigh, three thigh, "
            + "fused thigh, extra thigh, worst thigh, missing fingers, "
            + "extra fingers, ugly fingers, long fingers, horn, "
            + "extra eyes, huge eyes, 2girl, amputation, disconnected limbs"
    )
    negative_prompt = default_negative_prompt + (negative_prompt or "")

    generated_image = generate_image_by_sd(
        model="dreamshaperXL_v21TurboDPMSDE.safetensors [4496b36d48]",
        prompt=prompt, negative_prompt=negative_prompt,
        steps=6, cfg=2,
        sampler_name="DPM++ SDE", scheduler="Karras",
        width=1024, height=1024
    )
    character_name = character_name.replace(" ", "_")
    os.makedirs(f"characters/{character_name}", exist_ok=True)

    card_path = f"characters/{character_name}/{character_name}.png"

    generated_image.save(card_path)
    print("Generated character avatar")
    return generated_image


def input_none(text):
    user_input = text
    if user_input == "":
        return None
    else:
        return user_input


def export_as_json(
        name, summary, personality, scenario, greeting_message, example_messages
):
    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"
    os.makedirs(base_path, exist_ok=True)
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path="",
    )
    character_name = character.name.replace(" ", "_")
    json_path = f"{base_path}{character_name}.json"
    character.export_neutral_json_file(json_path)
    return character.export_neutral_json(), json_path


def export_character_card(
        name, summary, personality, scenario, greeting_message, example_messages
):
    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"
    os.makedirs(base_path, exist_ok=True)
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=f"{base_path}{character_name}.png",
    )
    character_name = character.name.replace(" ", "_")
    card_path = f"{base_path}{character_name}.card.png"
    character.export_neutral_card_file(card_path)
    return Image.open(card_path), card_path


def read_file_to_list_of_tuples(file_path: str) -> Optional[list[tuple[str, str]]]:
    try:
        result = []

        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Strip any surrounding whitespace and split by comma
                items = line.strip().split(',')
                # Convert the split line into tuple and append to the result list
                result.append((items[0], items[1]))
        if len(result) == 0:
            return None
        return result
    except Exception as e:
        print(f"read_file_to_list_of_tuples error: {e}")
        return None


def build_webui(host: str, port: int, auth_file: str):
    with gr.Blocks() as webui:
        gr.Markdown("# 角色工厂 WebUI")
        gr.Markdown(
            "## 用于生成 [SillyTavern](https://docs.sillytavern.app/) 的角色扮演配置，在公司内网访问 [SillyTavern](http://10.0.0.164:8000/)")
        with gr.Tab("创建角色"):
            gr.Markdown(
                "### 注意: 请按顺序，从上到下依次生成，每次生成后，可以人工修改校验后，再执行下一块。全部执行完后，切换到【导出角色】将角色导出"
            )
            model = gr.Dropdown(
                choices=[*supported_openai_models, *supported_ollama_models],
                value=supported_openai_models[0],
                multiselect=False,
                label="LLM Model", info="Generation LLM model"
            )
            topic = gr.Textbox(
                placeholder="Topic: The topic for character generation (e.g., Fantasy, Anime, etc.)",  # nopep8
                label="topic",
            )
            gender = gr.Textbox(
                placeholder="Gender: Gender of the character (e.g., male, female.)", label="gender"
            )
            with gr.Column():
                with gr.Row():
                    name = gr.Textbox(placeholder="character name", label="name")
                    name_button = gr.Button("Generate character name with LLM")
                    name_button.click(
                        generate_character_name,
                        inputs=[model, topic, gender],
                        outputs=name
                    )
                with gr.Row():
                    summary = gr.Textbox(
                        placeholder="character summary",
                        label="summary"
                    )
                    summary_button = gr.Button("Generate character summary with LLM")  # nopep8
                    summary_button.click(
                        generate_character_summary,
                        inputs=[model, name, topic, gender],
                        outputs=summary,
                    )
                with gr.Row():
                    personality = gr.Textbox(
                        placeholder="character personality", label="personality"
                    )
                    personality_button = gr.Button(
                        "Generate character personality with LLM"
                    )
                    personality_button.click(
                        generate_character_personality,
                        inputs=[model, name, summary, topic],
                        outputs=personality,
                    )
                with gr.Row():
                    scenario = gr.Textbox(
                        placeholder="character scenario",
                        label="scenario"
                    )
                    scenario_button = gr.Button("Generate character scenario with LLM")  # nopep8
                    scenario_button.click(
                        generate_character_scenario,
                        inputs=[model, summary, personality, topic],
                        outputs=scenario,
                    )
                with gr.Row():
                    greeting_message = gr.Textbox(
                        placeholder="character greeting message",
                        label="greeting message"
                    )
                    greeting_message_button = gr.Button(
                        "Generate character greeting message with LLM"
                    )
                    greeting_message_button.click(
                        generate_character_greeting_message,
                        inputs=[model, name, summary, personality, topic],
                        outputs=greeting_message,
                    )
                with gr.Row():
                    example_messages = gr.Textbox(
                        placeholder="character example messages",
                        label="example messages"
                    )
                    example_messages_button = gr.Button(
                        "Generate character example messages with LLM"
                    )
                    example_messages_button.click(
                        generate_example_messages,
                        inputs=[model, name, summary, personality, topic],
                        outputs=example_messages,
                    )
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(width=512, height=512)
                    with gr.Column():
                        negative_prompt = gr.Textbox(
                            placeholder="negative prompt for stable diffusion (optional)",  # nopep8
                            label="negative prompt",
                        )
                        avatar_prompt = gr.Textbox(
                            placeholder="prompt for generating character avatar (If not provided, LLM will generate prompt from character description)",
                            # nopep8
                            label="stable diffusion prompt",
                        )
                        avatar_button = gr.Button(
                            "Generate avatar with stable diffusion (set character name first)"  # nopep8
                        )
                        avatar_button.click(
                            generate_character_avatar,
                            inputs=[
                                model,
                                name,
                                summary,
                                topic,
                                negative_prompt,
                                avatar_prompt,
                            ],
                            outputs=image_input,
                        )
        with gr.Tab("导出角色"):
            with gr.Column():
                with gr.Row():
                    export_image = gr.Image(width=512, height=512, show_download_button=False)
                    export_json_textbox = gr.JSON()

                with gr.Row():
                    export_card_button = gr.DownloadButton("Export as character card")
                    export_json_button = gr.DownloadButton("Export as JSON")

                    export_card_button.click(
                        export_character_card,
                        inputs=[
                            name,
                            summary,
                            personality,
                            scenario,
                            greeting_message,
                            example_messages,
                        ],
                        outputs=[export_image, export_card_button],
                    )
                    export_json_button.click(
                        export_as_json,
                        inputs=[
                            name,
                            summary,
                            personality,
                            scenario,
                            greeting_message,
                            example_messages,
                        ],
                        outputs=[export_json_textbox, export_json_button],
                    )

    auths = read_file_to_list_of_tuples(auth_file)
    webui.launch(show_api=False, server_name=host, server_port=port, auth=auths)


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="Config running")

    # 添加参数
    parser.add_argument('--host', type=str, default="0.0.0.0", help='The host address')
    parser.add_argument('--port', type=int, default=7860, help='The port number (default: 7860)')
    parser.add_argument('--auth_file_path', type=str, default="", help='WebUI auth config file path')
    parser.add_argument('--sd_host', type=str, default="http://localhost:7860",
                        help='stable-diffusion-webui api url (default: http://localhost:7860)')
    parser.add_argument('--ollama_host', type=str, default="http://localhost:11434",
                        help='ollama api url (default: http://localhost:11434)')
    parser.add_argument('--openai_host', type=str, default="https://api.openai.com",
                        help='openai api url (default: https://api.openai.com)')
    parser.add_argument('--openai_key', type=str, required=True, help='openai api key')

    # 解析参数
    args = parser.parse_args()

    build_webui(args.host, args.port, args.auth_file_path)
