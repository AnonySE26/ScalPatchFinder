from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import re
import sys


client = OpenAI()
prompt_template="""
Given the CVE description of [software name], extract the spans of the exact phrase (variable, file, method, class, etc) in the code to help retrieve the patching commit of the vulnerability. If no span, output nothing. Do not include 'Output: ' in your response. Only output words in the Input. The span will be used for search within a GitHub repo, therefore only include camel/snake case, or spans that look similar to camel/snake e.g., connected using -, +, :, etc. Do not extract (1) words describing the software name/version; (2): words describing the vulnerability. Do not extract natural language spans or spans that look too general for search. \n\n 1. Input: The Double.parseDouble method in Java Runtime Environment (JRE) in Oracle Java SE and Java for Business 6 Update 23 and earlier, 5.0 Update 27 and earlier, and 1.4.2_29 and earlier, as used in OpenJDK, Apache, JBossweb, and other products, allows remote attackers to cause a denial of service via a crafted string that triggers an infinite loop of estimations during conversion to a double-precision binary floating-point number, as demonstrated using 2.2250738585072012e-308 \n    Output: Double.parseDouble. Wrong Output: a crafted string (natural language and vulnerability/exploit) \n\n 2. Input: A time-of-check-time-of-use race condition vulnerability in Buildkite Elastic CI for AWS versions prior to 6.7.1 and 5.22.5 allows the buildkite-agent user to bypass a symbolic link check for the PIPELINE_PATH variable in the fix-buildkite-agent-builds-permissions script. \n    Output: PIPELINE_PATH variable,fix-buildkite-agent-builds-permissions script \n\n 3. Input: The br_multicast_add_group function in net/bridge/br_multicast.c in the Linux kernel before 2.6.38, when a certain Ethernet bridge configuration is used, allows local users to cause a denial of service (memory corruption and system crash) by sending IGMP packets to a local interface. \n    Output: br_multicast_add_group, net/bridge/br_multicast.c, Wrong output: Ethernet bridge configuration (natural language span) IGMP (vulnerability/exploit) \n\n 4. Input: A memory leak in rsyslog before 5.7.6 was found in the way deamon processed log messages are logged when $RepeatedMsgReduction was enabled. A local attacker could use this flaw to cause a denial of the rsyslogd daemon service by crashing the service via a sequence of repeated log messages sent within short periods of time \n    Output: $RepeatedMsgReduction \n\n 5. Input: Apache Tomcat 8.5.0 to 8.5.63, 9.0.0-M1 to 9.0.43 and 10.0.0-M1 to 10.0.2 did not properly validate incoming TLS packets. When Tomcat was configured to use NIO+OpenSSL or NIO2+OpenSSL for TLS, a specially crafted packet could be used to trigger an infinite loop resulting in a denial of service. \n    Output: NIO+OpenSSL, NIO2+OpenSSL. Wrong Output: TLS (vulnerability words), denial of service (natural language span, vulnerability/exploit) \n\n 7. Input: Jenkins Active Choices Plugin 2.5.6 and earlier does not escape the parameter name of reactive parameters and dynamic reference parameters, resulting in a stored cross-site scripting (XSS) vulnerability exploitable by attackers with Job/Configure permission \n    Output: None. Wrong Output: reactive parameters (natural language words); Job/Configure (vulnerability/exploit). \n\n 8. Input: Rendertron 1.0.0 allows for alternative protocols such as 'file://' introducing a Local File Inclusion (LFI) bug where arbitrary files can be read by a remote attacker. \n    Output: None, Wrong output: file:// (too general), Local File Inclusion (LFI) (vulnerability/exploit) \n\n 9. Input: [cve_desc] \n    Output:
 """
# print(prompt_template)

dataset = sys.argv[1]
split = sys.argv[2]

input_file=f'../result/{dataset}_{split}_path_gpt.csv'
post_processed_output_file = f'../result/{dataset}_{split}_path.csv'

def generate_response(prompt_template, excel_content, model="gpt-4o"):
    prompt = prompt_template.replace("[cve_desc]", excel_content)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip()

df = pd.read_csv(input_file)
df['gpt'] = df.get('gpt', pd.Series(dtype=object))

try:
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        time.sleep(2)
        text = row['desc']
        if pd.notna(text):
            response = generate_response(prompt_template, excel_content=text)
            df.at[i, 'gpt'] = response

    df.to_csv(input_file, index=False)
except KeyboardInterrupt:
    print('KeyboardInterrupt detected, saving current progress...')
    df.to_csv(input_file, index=False)
except Exception as e:
    print(f"Error encountered: {e}, saving current progress...")
    df.to_csv(input_file, index=False)


def process_gpt(row):
    if pd.isna(row['gpt']):
        return ''

    tokens = [x.strip() for x in row['gpt'].split(',')]
    mylist = [token for token in tokens if not re.search(r"\s", token) and not re.match("^[a-z]+$", token)]

    return ', '.join(mylist)

df['gpt'] = df.apply(process_gpt, axis=1)
df.rename(columns={'gpt': 'file_func_name'}, inplace=True)
df.drop(columns=['desc'], inplace=True)
df.to_csv(post_processed_output_file, index=False)
print(f"\nFile saved as {post_processed_output_file}")
