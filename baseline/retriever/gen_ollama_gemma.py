import subprocess

def summarize_with_gemma(retrieved_passages, device_name):
    """Summarize retrieved reviews with Ollama gemma."""
    retrieved_passages = [p for p in retrieved_passages if p]  # Filter out empty
    if not retrieved_passages:
        print("⚡ Warning: No reviews to summarize.")
        return ""

    retrieved_text = " ".join(retrieved_passages)
    # prompt = f"""Summarize the following reviews into a clear and concise summary:\n{retrieved_text}\nSummary:"""   

    # prompt = f"""
    #             Summarize the following user reviews about {device_name} only, focusing on user opinions about the device’s features, pros and cons. Do not include information about other phones.

    #             Reviews:
    #             {retrieved_text}

    #             Summary:
    #             """

    # prompt = f"""
    # You are a helpful assistant. Summarize ONLY the following user reviews about {device_name}.  
    # Please provide a concise, consistent summary in the following format:

    # Pros:
    # - [list pros here]

    # Cons:
    # - [list cons here]

    # Summary:
    # - [one to two sentence summary]

    # Reviews:
    # {retrieved_text}

    # Summary:
    # """

    prompt = f"""
            You are a helpful assistant. Summarize ONLY the following user reviews about {device_name}.

            Please write a concise summary in exactly 5 sentences, covering pros and cons equally.  
            Use simple, direct language with no fluff or repetition.  
            Do NOT mention any other phones or brands.

            Reviews:
            {retrieved_text}

            Summary:
            """

    try:
        result = subprocess.check_output(
            ['/usr/local/bin/ollama', 'run', 'gemma2:2b'],
            input=prompt.encode('utf-8'),
            stderr=subprocess.STDOUT
        )
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        print("Error while calling Ollama.", e.output.decode('utf-8'))
        return "Error generating summary."
