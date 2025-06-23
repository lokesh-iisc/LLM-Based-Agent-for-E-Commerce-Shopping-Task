import requests
import json
from bs4 import BeautifulSoup
import uuid
from typing import List, Dict
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# Load your fine-tuned model and tokenizer
# Path to your fine-tuned checkpoint
MODEL_PATH = "/home/lokeshk/webshop/qwen_fine_tune_model_second/checkpoint-8000"

# 1. Load tokenizer (with special tokens)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only = True
)
import os
chat_template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
if os.path.exists(chat_template_path):
    with open(chat_template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()
else:
    raise FileNotFoundError(f"Missing chat_template.jinja at {chat_template_path}")
SPECIAL_TOKENS = ["<|sep|>"]
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
# ensure pad_token exists
tokenizer.pad_token = tokenizer.eos_token

# 2. Load model in bfloat16 as during training
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
    local_files_only = True
)
# resize embeddings to include <|sep|>
model.resize_token_embeddings(len(tokenizer))
model.eval()


class SimpleAgent:
    def __init__(self, base_url, model, tokenizer):
        self.base_url = base_url
        self.session_id = None
        self.score = 0
        self.model = model
        self.tokenizer = tokenizer

    def start_session(self):
        self.session_id = str(uuid.uuid4())

        response = requests.get(f"{self.base_url}/{self.session_id}")
        if response.status_code == 200:
            #print(f"Started session with ID: {self.session_id}")
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            instruction_element = soup.find('div', id = 'instruction-text')
            #print("instruction_element",instruction_element)
            if instruction_element:
                instruction = instruction_element.text.strip()
                if instruction.lower().startswith("instruction:"):
                    instruction = instruction[len("Instruction:"):].strip()
                    #print(f"Extracted Instruction: {instruction}")
                    return instruction
                
        else:
            print(f"Failed to initialize session with ID: {self.session_id}. Server responded with: {response.status_code}")


    def search(self, query):
    
        keywords = query.lower().split(" ")
        
        response = requests.get(
            f"{self.base_url}/search_results/{self.session_id}/{keywords}/1"
        )

        if response.status_code != 200:
            print(f"Search request failed: {response.status_code}")
            return []

        try:
            html = response.text
            print(f"Search results for '{query}': Retrieved HTML")
            return self.parse_products_from_html(html)
        except Exception as e:
            print(f"Error parsing search results: {e}")
            return []

    

    def parse_products_from_html(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        products = []

        product_elements = soup.find_all('div', class_='list-group-item')
        #print('product_element:',product_elements)

        for element in product_elements:
            try:
                asin = element.find('h4', class_='product-asin').a.text.strip()

                title = element.find('h4', class_='product-title').text.strip()

                try:
                    price_text = element.find('h5', class_='product-price').text.strip()
                    if "to" in price_text:
                        lower, upper = map(lambda x: float(x.replace('$', '').replace(',', '')), price_text.split("to"))
                        price = (lower + upper) / 2
                    else:
                        price = float(price_text.replace('$', '').replace(',', ''))
                except ValueError:
                    print(f"Error parsing price: {price_text}")
                    price = None

                image_url = element.find('img', class_='result-img')['src']

                products.append({
                    "asin": asin,
                    "name": title,
                    "price": price,
                    "image": image_url,
                })
            except AttributeError as e:
                print(f"Error parsing product element: {e}")
        #print('Products1',products)
        return products

    def view_product(self, asin, query, page, options):

        response = requests.get(
            f"{self.base_url}/item_page/{self.session_id}/{asin}/{query}/{page}/{options}"
        )
        if response.status_code != 200:
            print(f"Failed to view product: {response.status_code}")
            return None

        html = response.text
        print(f"Viewed product details for ASIN: {asin}")
        return self.parse_product_details_from_html(html)

    def parse_product_details_from_html(self, html):
    
        soup = BeautifulSoup(html, 'html.parser')
        try:
            image_url = soup.find('img', id='product-image')['src']

            title = soup.find('h2').text.strip()

            try:
                price_text = soup.find('h4', text=lambda x: x and "Price:" in x).text.strip()
                price_text = price_text.replace('Price:', '').strip()
                if "to" in price_text:
                    lower, upper = map(lambda x: float(x.replace('$', '').replace(',', '')), price_text.split("to"))
                    price = (lower + upper) / 2
                else:
                    price = float(price_text.replace('$', '').replace(',', ''))
            except ValueError:
                print(f"Error parsing price: {price_text}")
                price = None

            try:
                rating_text = soup.find('h4', text=lambda x: x and "Rating:" in x).text.strip()
                rating = float(rating_text.replace('Rating:', '').strip())
            except (ValueError, AttributeError):
                print(f"Error parsing rating: {rating_text}")
                rating = None  


            options = {}
            for option_section in soup.find_all('div', class_='radio-toolbar'):
                option_name = option_section.find_previous('h4').text.strip()
                option_values = [
                    label.text.strip()
                    for label in option_section.find_all('label')
                ]
                options[option_name] = option_values

            return {
                "image_url": image_url,
                "title": title,
                "price": price,
                "rating": rating,
                "options": options,
            }
        except AttributeError as e:
            print(f"Error parsing product details: {e}")
            return None

        
    def view_item_sub_page(self, asin, query, page, sub_page, options):
        
        response = requests.get(
            f"{self.base_url}/item_sub_page/{self.session_id}/{asin}/{query}/{page}/{sub_page}/{options}"
        )
        if response.status_code != 200:
            print(f"Failed to view sub-page: {response.status_code}")
            return None

        html = response.text
        print(f"Viewed sub-page {sub_page} for ASIN: {asin}")

        if sub_page == "Attributes":
            return self.parse_attributes_page(html)
        elif sub_page == "Features":
            return self.parse_features_page(html)
        elif sub_page == "Reviews":
            return self.parse_review_page(html)
        elif sub_page == "Description":
            return self.parse_description_page(html)
        else:
            print(f"Unknown sub-page type: {sub_page}")
            return None

    def parse_attributes_page(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        try:
            attributes = [
                attribute.text.strip()
                for attribute in soup.find_all('p', class_='attribute')
            ]
            category = soup.find('h5', class_='product-category').text.strip()
            query = soup.find('h5', class_='product-query').text.strip()
            product_category = soup.find('h5', class_='product-product_category').text.strip()

            return {
                "type": "Attributes",
                "attributes": attributes,
                "category": category,
                "query": query,
                "product_category": product_category
            }
        except AttributeError as e:
            print(f"Error parsing attributes page: {e}")
            return None
        
    def parse_features_page(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        try:
            features = [
                feature.text.strip()
                for feature in soup.find_all('p', class_='product-info')
            ]
            return {"type": "Features", "features": features}
        except AttributeError as e:
            print(f"Error parsing features page: {e}")
            return None


    def parse_review_page(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        try:
            reviews = []
            for review_card in soup.find_all('div', class_='card'):
                title = review_card.find('h4', class_='blue-text').text.strip()
                score = int(review_card.find('span').text.strip())
                body = review_card.find('p', class_='content').text.strip()
                reviews.append({"title": title, "score": score, "body": body})

            return {"type": "Reviews", "reviews": reviews}
        except AttributeError as e:
            print(f"Error parsing review page: {e}")
            return None

    def parse_description_page(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        try:
            description = soup.find('p', class_='product-info').text.strip()
            return {"type": "Description", "description": description}
        except AttributeError as e:
            print(f"Error parsing description page: {e}")
            return None

    

    def buy_product(self, asin, options):
        
        response = requests.get(
            f"{self.base_url}/done/{self.session_id}/{asin}/{options}"
        )
        if response.status_code == 200:
            html = response.text
            print(f"Bought product with ASIN: {asin}")
            return self.parse_done_page(html)
        else:
            print(f"Failed to buy product with ASIN: {asin}")
            return None
        
    def parse_done_page(self, html):
        
        soup = BeautifulSoup(html, 'html.parser')
        try:
            
            thank_you_message = soup.find('h1', id='thankyou')
            thank_you_message = thank_you_message.text.strip() if thank_you_message else "Thank you!"

            mturk_code = soup.find('pre')
            mturk_code = mturk_code.text.strip() if mturk_code else "N/A"

            reward_text = soup.find('h3', id='reward')
            reward_text = reward_text.find('pre').text.strip() if reward_text and reward_text.find('pre') else "0"
            reward = float(reward_text) if reward_text.replace('.', '', 1).isdigit() else 0

            asin = soup.find('h4', id='asin')
            asin = asin.find('pre').text.strip() if asin and asin.find('pre') else "Unknown"

            options = soup.find('h4', id='options')
            options = options.find('pre').text.strip() if options and options.find('pre') else "{}"

            purchased_attrs = soup.find('h4', id='purchased_attrs')
            purchased_attrs = purchased_attrs.find('pre').text.strip() if purchased_attrs and purchased_attrs.find('pre') else "N/A"

            category = soup.find('h4', id='purchased-category')
            category = category.find('pre').text.strip() if category and category.find('pre') else "N/A"

            query = soup.find('h4', id='purchased-query')
            query = query.find('pre').text.strip() if query and query.find('pre') else "N/A"

            product_category = soup.find('h4', id='purchased-pc')
            product_category = product_category.find('pre').text.strip() if product_category and product_category.find('pre') else "N/A"

            return {
                "thank_you_message": thank_you_message,
                "mturk_code": mturk_code,
                "reward": reward,
                "product_details": {
                    "asin": asin,
                    "options": options,
                    "purchased_attrs": purchased_attrs,
                    "category": category,
                    "query": query,
                    "product_category": product_category,
                }
            }
        except Exception as e:
            print(f"Error parsing done page: {e}")
            return None
        
    def model_chat(self, prompt):
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False) + "<|im_start|>assistant\n"
        #print("Prompt text:\n", prompt_text)
        encoding = self.tokenizer(prompt_text,return_tensors="pt",padding=False, add_special_tokens=False).to(self.model.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        #print("Decoded response:", repr(response))
        return response

    def generate_query(self, instruction):
        prompt_query = [
            {"role": "system", "content": "You are a helpful assistant that writes concise e-commerce search queries."},
            {"role": "user", "content": (
                f"Instruction: {instruction}\n\n"
                "Now generate only the search query. Do not include any extra explanation or formatting. "
                "Just write the query as a single line."
            )}
        ]
        query = self.model_chat(prompt_query).strip()
        query = query.replace("|", "").strip()

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1].strip()

        print("Query:", query)

        return query

    def select_with_model(self,
        instruction: str,
        query: str,
        top_products: List[Dict],   
    ) -> int:
        # 1. build the product block
        lines = []
        for i, p in enumerate(top_products, start=1):
            price_txt = f" – ${float(p.get('price',0)):.2f}" if p.get("price") else ""
            title    = p["name"].splitlines()[0]
            lines.append(f"{i}. {title}{price_txt}")
        product_block = "\n".join(lines)

        # 2. build the prompt
        prompt_select = [
            {"role": "system", "content": "Pick the best product that matches the instruction."},
            {"role": "user", "content": f"""
        Instruction: {instruction}
        Search Query: {query}

        Here are the top-10 search results:
        {product_block}

        Please respond with **only a single number from 1 to 10**, representing the best matching product. Do not explain or include any other text. Just write the number.
        """}
        ]
        selection = self.model_chat(prompt_select)
        match = re.search(r"\b([1-9]|10)\b", selection)
        if match:
            idx = int(match.group(1)) - 1
        else:
            print("Invalid index, defaulting to 0. Raw output:", selection)
            idx = 0
        print("Selected index:", idx)
        return idx

    def run(self, num_run):
        
        results = []

        for _ in range(num_run):
            instr = self.start_session()
            print('Instruction :', instr)
            query = self.generate_query(instr)
            print("Query",query)
            products = self.search(query)
            #print('Products2',products)
            if products:
                top_10 = products[:10]
                selection = self.select_with_model(instr,query,top_10)
                print("selection",selection)
                choosen_product = top_10[selection] 
                asin = choosen_product["asin"]
                product_name = choosen_product.get("name", "Unknown Product")
                product_price = choosen_product.get("price", "N/A")
                options = choosen_product.get("options", {})  
                print(f"Selected product: {product_name} (ASIN: {asin}, Price: {product_price})")
                print(f"Options: {options}")

                product_details = self.view_product(asin, query, 1,options)
                if product_details:
                    options = product_details.get("options", {})
                    print("option",options)
                    #print(f"Product details: {product_details}")

                    sub_page = "Description"  
                    sub_page_details = self.view_item_sub_page(asin, query, 1, sub_page, options)
                    #print(f"Sub-page details ({sub_page}): {sub_page_details}")

                    print(f"Attempting to buy product: {product_name} (ASIN: {asin}) with options: {options}")
                    purchase_result = self.buy_product(asin, options)
                    if purchase_result:
                        print(f"Purchase successful: {purchase_result}")
                        reward = purchase_result['reward']
                        print(f"reward : {reward}")
                    else:
                        print(f"Purchase failed for product: {product_name} (ASIN: {asin})")

                    results.append({
                        "Instr": instr,
                        "query": query,
                        "Bought_product": product_name,
                        "product_detail": purchase_result["product_details"] if purchase_result else {},
                        "reward": reward
                    })
        with open("results1.json", "w") as f:
            json.dump(results, f, indent=4)            
        print("Agent finished all tasks.")

if __name__ == "__main__":
    agent = SimpleAgent(base_url="http://10.32.50.50:3000", model = model, tokenizer=tokenizer)  
    num_run = 500
    agent.run(num_run)