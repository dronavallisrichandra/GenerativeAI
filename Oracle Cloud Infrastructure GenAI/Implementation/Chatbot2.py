# coding: utf-8
# Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

##########################################################################
# generate_text_demo.py
# Supports Python 3
##########################################################################
# Info:
# Get texts from LLM model for given prompts using OCI Generative AI Service.
##########################################################################
# Application Command line(no parameter needed)
# python generate_text_demo.py
##########################################################################
import oci

# Setup basic variables
# Auth Config
# TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
compartment_id = "ocid1.tenancy.oc1..aaaaaaaaf4a2mihhz26covfccz5yqnm2c4fr6qgwxoexfu7xadgf5h4zsvoq"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
generate_text_detail = oci.generative_ai_inference.models.GenerateTextDetails()
llm_inference_request = oci.generative_ai_inference.models.CohereLlmInferenceRequest()
llm_inference_request.prompt = "{input}"
llm_inference_request.max_tokens = 600
llm_inference_request.temperature = 1
llm_inference_request.frequency_penalty = 0
llm_inference_request.top_p = 0.75

generate_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyafhwal37hxwylnpbcncidimbwteff4xha77n5xz4m7p6a")
generate_text_detail.inference_request = llm_inference_request
generate_text_detail.compartment_id = compartment_id
generate_text_response = generative_ai_inference_client.generate_text(generate_text_detail)
# Print result
print("**************************Generate Texts Result**************************")
print(generate_text_response.data)
