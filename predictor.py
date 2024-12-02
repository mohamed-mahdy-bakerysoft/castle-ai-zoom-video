from openai import OpenAI
import anthropic
import json


class Predictor:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key        


class GPTAdapter(Predictor):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        self.client =OpenAI(api_key=api_key,)
        

    def predict(self, model_name, input_text):
        response = self.client.predict(model_name, input_text)
        return response
    
        

class ClaudeAdapter(Predictor):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        print(self.api_key)
        self.client = anthropic.Anthropic(api_key = self.api_key)
        self.system_prompt = """
                You are an intelligent assistant who helps to identify zoom-in moments in a video transcript.
                
                You have the transcript in a specific format:
                - **Capitalized words** indicate emphasis in the audio.
                - **Pauses** are denoted by brackets, e.g., `[...s]` indicating a number of seconds.
                - **Start and End Times** of each sentence are provided at the end of each sentence.

                # Zoom-in Key Indicators (Ranked by Descending Priority)
                1. **Keywords/Phrases**: Look for capitalized words or phrases that signify major importance, especially if they are **followed by a silence** or pause.
                2. **Important Concepts**: Phrases beginning with conjunctions such as “but...“, “and...“, “so...“, or “if...“.
                3. **Emotional Questions**: Questions that convey strong emotion or enthusiasm.
                4. **Exclamations**: Statements like “This is incredible!” that carry exclamation.
                5. **General Emphasized Words**: Capitalized words or phrases not followed by a silence.

                
                # Rules for Zoom-ins and Jump Cuts
                - **Zoom-ins** should be followed by a jump cut, placed **at the end of a sentence or idea**. However, ensure at least a **3-second interval** between the zoom-in completion and the jump cut.
                - Avoid overlapping zoom-ins; select the **most impactful moment** if key indicators are detected closely together.
                - Ensure the chosen jump cuts do not interfere with subsequent zoom-ins and adequate spacing.

                # CRITICAL TIMING RULE
                - You MUST select ONE zoom-in moment per minute of video duration
                - For example:
                - 5-minute video = approximately 5 zoom-in moments
                - 10-minute video = approximately 10 zoom-in moments
                - ENSURE at least a **3-second interval** between the zoom-in completion and the jump cut
                - Your output should strictly follow this 1-per-minute rule regardless of how many good candidates you find
       
                 # Steps

                1. **Identify Key Areas for Zoom-in**:
                - Analyze the transcript and find potential zoom-in points based on the priority indicators.
                - Determine the emphasis and contextual importance of key moments before deciding zoom-in placement.
        
                2. **Select the Optimal Zoom-in Point**:
                - If multiple zoom-in points occur close together, prioritize the **most significant moment** using the ranking above.
                - Avoid using too many zoom-ins in succession to avoid disorienting the viewer.
                - Ensure zoom-ins are equally distributed across the video to maintain a balanced pacing. Avoid clustering zoom-ins in one section while leaving others sparse.
        
                3. **Identify Transition Points**:
                - Determine a suitable jump cut for each zoom-in — cue it at the **end of the sentence or idea**.
                - If the zoom in is in the end of sentence, don’t select next sentence start as a jumpcut, find more appropriate moment(the start of the sentence 2 sentence away or when the idea ends), cause the 3 second timing is MUST 
                - Ensure transitions align naturally to avoid interrupting important content flow.
        
                4. **Verify Final Selections**:
                - Check that zoom-ins do not overlap and balance emotional intensity or conceptual focus.
                - Verify all transitions are smooth and do not interfere with the following zoom-in or overall narrative.

                # Output Format

                Return a JSON object with these fields: {“zoom_moments”: [{fields}]}
                - **sentence_number**: The sentence number where the zoom-in occurs.
                - **zoom_in_phrase**: The specific word/phrase exactly as written in the transcript where the zoom-in starts.
                - **reason**: Why this keyword/phrase was chosen (explain based on provided priorities).
                - **transition_sentence_number**: The sentence number where the jump cut transition occurs.
                - **transition_sentence_word**: The exact word/phrase exactly as written in the transcript where the jump cut begins.
                - **transition_reason**: Explanation for why this word marks the best transition point.

                # Examples

                ### Example 1
                - **Sentence Number**: 28
                - **Zoom-in Phrase**: “but this CHANGES EVERYTHING”
                - **Reason**: Important concept signified by conjunction “but” followed by an emphasized phrase.
                - **Transition Sentence Number**: 29
                - **Transition Sentence Word**: “Let’s”
                - **Transition Reason**: Start of the next sentence, marks a shift in direction, providing time for the idea to sink in before moving forward.
        
                ### Example 2
                - **Sentence Number**: 15
                - **Zoom-in Phrase**: “SUCCESS”
                - **Reason**: High emphasis word in capital letters followed by a 2-second pause indicating significance.
                - **Transition Sentence Number**: 17
                - **Transition Sentence Word**: “concluded”
                - **Transition Reason**: Marks the end of the idea in sentence 15, ensures a natural flow to the next sentence.
        
        """
        self.prompt =  """ Analyze the provided video transcript to determine optimal placements for fast zoom-ins based on the given priority indicators."""
        

    def preprocess_input(inputs):
        return ''.join(inputs)
    
    def extract_json(self, response):
        json_start = response.index("{")
        json_end = response.rfind("}")
        return json.loads(response[json_start:json_end+1])
            
    def get_predictions(self, inputs, num_inputs):
        predictions = []
        for inp in inputs[:num_inputs]:
            preprocessed_input = self.preprocess_input(inp)
            prompt_ = self.prompt + '\n' + preprocessed_input
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_
                            }
                        ]
                    }
                ]
                )
            out = self.extract_json(message.content[0].text)
            predictions.append(out)

            
        return predictions
    
        
    
