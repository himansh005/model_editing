import requests
import json
import re
import time

from bs4 import BeautifulSoup

def parse_html_results(html_content):
    """
    Parse the results from the given HTML content.

    Parameters:
    - html_content (str): The HTML content to parse.

    Returns:
    - list: Parsed items.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the div with class 'w2vresultblock'
    result_div = soup.find('div', class_='w2vresultblock')

    if result_div:
        # Extract text content and split into a list of items
        items = result_div.get_text(separator='\n').strip().split('\n')
        items = [item for item in items if item.strip()]
        return items
    else:
        print("No div with class 'w2vresultblock' found in the HTML.")
        return []

def get_w2v_neighbours(word, k):
    
    url = "http://epsilon-it.utu.fi/wv_demo/nearest"  # Replace with the actual API endpoint URL

    payload = {
        'form[0][name]': 'word',
        'form[0][value]': word,
        'form[1][name]': 'topn',
        'form[1][value]': str(k),
        'model_name': 'English GoogleNews Negative300'
    }

    response = requests.post(url, data=payload)


    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            tbl_data = data.get("tbl", "")
            data = parse_html_results(tbl_data)
            return data
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)  # This will print the error message or additional information

def get_facts(words, K):
    
    prompt_prepend = """Your task is to generate  single sentence facts about a given WORD.

    IMPORTANT: make sure facts are of form:

            {
            "stem": any fact that includes the target_word.
            "object": a person, place or thing, etc, but not a general noun that is a natural continuation of the stem.
            }

            Example: {
                "stem": "The Parisian skyline is adorned with iconic landmarks such as the ",
                "object": "Eiffel Tower"
            },
    1. The WORD should always be present in the stem and never be present in the object.
    2. Generate complete facts and avoid half sentences.
    
    Return the output as JSON file.
    """
    
    prompt = """
        WORD: {}
        K: {}
    """
    
    prompt = prompt.format(words, K)
    from pyChatGPT import ChatGPT

    session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..IYO8BeeM9WI9V4dJ.7pJVGPqcv3D6qLWE7VnC46kwF6hIDyg-t2rcsi1uPdzZWGWVomI7fcV_5M3iOeoBzdDl5iR8xNHgCZ4UHLPMyljtM2MtJh6TK9-t0hA1qbUpBIjt0MQO1IyUS9eICRu7xpqSn6BL3Vdb05LEMPtrQvwWvzLjCcp2SslMSMtHFJqmx-Lm-i8ZBQL8Wj091u_TSj4j2k3tJqMrK-KdRDOjGOWUE_IBWqLxK1HZbWe40jD7NLpMm7x6E9RQZI3fIcpXsHEWXT3DESwfko_EFyddeRLyZFMM9vTjVXn4DUj_62u4nnn4Gs3TZML1rew3fYDAchrMNoGCzkwBmViEfcuDu5w8WsSuaBNcK9pSJ0TR6JMzYsZXiaxDmlS88GIH33LH_QklOBESKCWF8-cOYHJDB7XOdKiKdrEpF-tPLgiTF1TlEUAGwwgG0cUuxP2a8tjyeQrssW5J-ZZOVt0k4YgEaQE6KkVI4QL_Tf7na24aMUhv3kSA2hM0M0xKogsRXrx4clfdkX89GVH0ymSmPqQFThG0faU783aF05yHKihYPFJJ58PKBJI13wEa1lX318xccyoAI_k-oxKWy5Hehoh0JhivjLX1kqCCnqArBQpuQ_bSVsWjAtqnV5imw9xLug9oGhcXH1iribCOME2Zzr-Vf41ZvmF_fuTZhWHcm7gFRAU1O4UnMRm1DSvDkKEQ6EJCOGOHE7ZezNIWbIKLLx-zvfo6qnSDQSnc9bWlYLG1lmaJLL5tSQWioIKEFarWJkvzWgi8HpFS6jmFOs3c48wUKfGzw_JiLZ0WB2D5aCB6x_y8XQr9mKPCtNb-7-C2Sw8kR8VhNOGNzDYQnqOWdqi8P36-nj_uQc9ZKAnzMmdGkKb3LHo6jjs3BYpoxi86ThrN3KxyS_oCHGlqQmppHXAJ1fUAOSrui6Slqd6UFkVunmFWVK8-snXajZ3BQZHA3KLkgAtuM_3C5VbKbySTNXGZclUAsBTDMMQG9CKNs6tnGuJULEqkiTmqM-EXznJaGAyg_P1k_XZhGOMAOfDjD0rQjjIsUrC5ZJhK4aRKfwcO8F2w71UwAoRBZAi9faAs1GGN5fd6X4KAN6Bd8G6G8TGQK1zx9tFnuE5aEyfTK-qjbMNUuOs_uS7ZYjHqmJeYTM24rbVe7NqrZsVMHeSRTzyBQnvZZgJioDHk6al3BEJbJd8lXQ89poCOe8Tfr0Vk1lyrGPD4ojg5h0t2DcFCLvwsiYn1a-3vXw_y8LIZul9nfV0e4dhzGBw5K-o7MzFbeZxC24GUoQWnzsnE_JkaF9Ft5GhXjyZvSADONtAZQL2Wn9NXrhPstbJ_JHpEgq8yXJQFGBr5YFs4ZgpEQJqxEGSu2l4DzX3_UNJK8nfsKU_LZfbhV0Shrq6nfQtf8ejnUTVHY2tztxP90e5XocFKWzpDvPFtXCy1ougc7O2ly9KfHTqPYo8MgDZ84CzVTcGlMusecM6-hTxa-7acx392cDGidZoGgTHILctnlbJIBHFpwefanTxqciwOmRsdY8gHerk_2iHds8fRBYllOqucMm9qvWyBAhk6XTEM7PcHNmGVJrYg8bCaw-aYYr15Vh2FOZHoEF21jyqo3g3vmDfJ63iZjYnEad11_IFMZocWNVNRTGvm9ucM4WEhX6uusrLiovhjLt3kYqUluMPS5DnobzLTILzZn56_xQhLJ7U7UvwLpYzyILdwV0-m8BvIWBLBJTGaz8WBvb4VfpcxV8zMhafsqYxUrHnUIfCyEjf1IAfhpDU2CCprnT2n0KGKvxRG4Bekyj8QyOPlDJ8Vo1k2oly1pxNPL-zQh1vgTwPQpwLonfGsQHZFg0MVAjr5rHXSRH03H1a7O50hb9bIXhWWdv0jg5BJ-MMmC2rrFDxJ97xLtoaWqqwzrOjrybD9Q72RM41Ci9w4HBZf_DuRN_GKADUTRqJa5RuXmifTqN8veM8meA-tPs7BgZRxIaxiBTIhYl85JjZyyuynZ1n1lr2OV8luhLMTIiMCWZ-5o6b83Yz-Dq8ep07ib2ursnbw133RemLFetU17eW2W8vxD078wZHJDVpC6Di7IrAnvZYzGL_nnyCQXESHMnoAdB2l0NzSsbkdZmbYXFtdcNUYolX6fLV7Ji9EQ8W244lGboCrNXLBklvkxnoUaxD489OaQzgbxsTF8eTkZMOUwivmMtnmnnllvqw8OMmDP6CyU_ONMg_nMVRI9lS-FeMZIcxY-HamYTOIij1QM8LTv9WU0mjoVfALpmX9PBkbZ6HgUApTb9La1mH-Pd4pb6n-xBkmOmoU19ozT7BG1g_XDgUQVJR11NB_S2_wBUKyspA_rFRl9GyvOjffOzYuFiLHPIxx7OPoLdBxLAZzIZguBcQTrEkFLD8dWd0WlkoqlnAcxEofs_5sIo-9g85gd-mEfZ_KjSkoxWmhoGiOiy5FhvZmdNh4B9DpNoI-IUVN0YW6id6du__uH8t61bHBUaql1x9DGCG4mPpsjLegep95HKb-V1OcMR6dPd6LI0AT9ybjvuktru86ReqgCARCiA_a9cHH2qntJwfklLX2qWjEiymK36ovg71n2Qi5fI7tMgWzpdhR8G0GUDsx_Tdq5hL4xFA5h9U5gbOoKjgXUe6mqAPqwAMe2TMrXgXFTew-n5O9xBA1vFgby-84ui3OIkNzBz4gSdtV2latswx2KG8jRHuCZq-oV8rpOt2xaTVESmMGTB2bi2kseN2JIM_CE7xZyA1NZg3346gFNGMbb9A56WzdC3Kb3SG4bHFHHxoB5BM3cEqv7GLibKnNfKByf8vf-b1sFxz_PtH_SpfYL0AINb2-ER2i.FYxlNAQ2NJVWrcH4ytGXSw'
    api = ChatGPT(session_token, conversation_id='e4377b94-a5f7-4447-b48c-b6106aeb006f')
    resp = api.send_message(prompt)
    
    json_pattern = re.compile(r'json`(.*?)`', re.DOTALL)
    
    json_match = json_pattern.search(resp["message"])

    for group in json_match.groups():
        json_content = group
        try:
            data = json.loads(json.dumps(json_content))
            print(data)
        except:
            print(resp)
            continue
        return data


def get_one_hop_facts(word, root, num_neighbours, depth, max_depth, num_facts, visited):
    
    data = []
    if depth >= max_depth:
        return data, visited

    neighbours = get_w2v_neighbours(word, num_neighbours)
    done = [] #["manhattan", "brooklyn", "midtown_manhattan", "greenwich_village", "new_york", "ny", "nj", "nyc"]

    for neighbour in neighbours:
        
        if neighbour.lower() == word.lower() or neighbour.lower() in done or neighbour==root.lower() or neighbour.lower() in visited:
            continue
        
        facts = []#get_facts(neighbour, num_facts)
        
        record = {
            "name":neighbour,
            "depth": depth+1,
        }
        visited.add(neighbour.lower())
        data.append(record)
        datas, visited = get_one_hop_facts(neighbour, root, num_neighbours, depth+1, max_depth, num_facts, visited)
        data += datas
        
    return data, visited
    
if __name__=="__main__":
    
        
    word = "Oslo"
    depth = 5
    num_neighbours = 10
    num_facts = 5
        
    records, visited = get_one_hop_facts(word, word, num_neighbours, 0, depth, num_facts, set())
    items = []
    visited = list(visited)
    for i in range(len(list(visited))):
        items.append(visited[i].title())
    
    with open("nbrs.json", "w") as f:
        json.dump(records, f)
    
    batch = 5
    for i in range(0, len(items), batch):
        ents = items[i:min(len(items), i+batch)]
        d = get_facts(ents, num_facts)
        with open("factd/facts_"+str(i)+".json","w") as f:
            json.dump(d, f)
        