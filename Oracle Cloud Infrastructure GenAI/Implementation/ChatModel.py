# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# chat_demo.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model for given prompts using OCI Generative AI Service.
##########################################################################
# Application Command line(no parameter needed)
# python chat_demo.py
##########################################################################
import oci

# Setup basic variables
# Auth Config
# TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
compartment_id = "ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
chat_detail = oci.generative_ai_inference.models.ChatDetails()

chat_request = oci.generative_ai_inference.models.CohereChatRequest()
chat_request.message = "Generate a job description for a data visualization expert with the following three qualifications only:\n1) At least 5 years of data visualization expert\n2) A great eye for details\n3) Ability to create original visualizations"
chat_request.max_tokens = 200
chat_request.temperature = 0
chat_request.frequency_penalty = 1
chat_request.top_p = 0.75
chat_request.top_k = 0


chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyawk6mgunzodenakhkuwxanvt6wo3jcpf72ln52dymk4wq")
chat_detail.chat_request = chat_request
chat_detail.compartment_id = compartment_id
chat_response = generative_ai_inference_client.chat(chat_detail)
# Print result
print("**************************Chat Result**************************")
print(vars(chat_response))
