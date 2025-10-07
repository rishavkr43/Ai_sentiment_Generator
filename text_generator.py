# text_generator.py
"""
Text Generation Module
Uses Hugging Face transformers for sentiment-aligned text generation
"""

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

class TextGenerator:
    """
    Generates text based on sentiment using GPT-2 and sentiment-specific prompting
    """
    
    def __init__(self, model_name='gpt2'):
        """
        Initialize the text generator with GPT-2 model
        
        Args:
            model_name (str): Name of the model to use (default: 'gpt2')
        """
        try:
            # Initialize the text generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_loaded = True
            # Store tokenizer for reference
            self.tokenizer = self.generator.tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            # Fallback tokenizer for basic operations
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
    
    def _get_sentiment_prompt(self, base_prompt, sentiment):
        """
        Create a contextually relevant prompt that continues the user's input
        with appropriate sentiment alignment
        
        Args:
            base_prompt (str): Original user prompt
            sentiment (str): Detected sentiment
            
        Returns:
            str: Enhanced prompt with sentiment context
        """
        # Clean the prompt
        base_prompt = base_prompt.strip()
        if not base_prompt.endswith(('.', '!', '?')):
            base_prompt += '.'
        
        # Create continuation prompts that naturally extend the user's input
        sentiment_templates = {
            'positive': [
                f"Story: {base_prompt} The atmosphere was electric with excitement.",
                f"Story: {base_prompt} Everyone shared in the joy of the moment.",
                f"Story: {base_prompt} This was a moment to celebrate.",
                f"Story: {base_prompt} The feeling of accomplishment was overwhelming.",
                f"Story: {base_prompt} Success felt incredible.",
            ],
            'negative': [
                f"Story: {base_prompt} The frustration was overwhelming.",
                f"Story: {base_prompt} This was incredibly disappointing.",
                f"Story: {base_prompt} Nothing seemed to go as planned.",
                f"Story: {base_prompt} The situation felt hopeless.",
                f"Story: {base_prompt} It was exhausting and demoralizing.",
            ],
            'neutral': [
                f"Story: {base_prompt} It was part of the daily routine.",
                f"Story: {base_prompt} The day continued as usual.",
                f"Story: {base_prompt} Nothing particularly noteworthy followed.",
                f"Story: {base_prompt} It was a typical occurrence.",
                f"Story: {base_prompt} The routine proceeded normally.",
            ]
        }
        
        # Select a random template for variety
        templates = sentiment_templates.get(sentiment, sentiment_templates['neutral'])
        return random.choice(templates)
    
    def generate_text(self, prompt, sentiment):
        """
        Generate text based on prompt and sentiment
        
        Args:
            prompt (str): User input prompt
            sentiment (str): Detected sentiment ('positive', 'negative', 'neutral')
            
        Returns:
            str: Generated text aligned with sentiment
        """
        if not self.model_loaded:
            # Use contextual fallback if model fails to load
            return self._generate_contextual_fallback(prompt, sentiment)
        
        # Enhance prompt with sentiment context
        enhanced_prompt = self._get_sentiment_prompt(prompt, sentiment)
        
        try:
            # Adjust generation parameters based on sentiment
            if sentiment == 'positive':
                temperature = 0.8  # More creative for positive
                top_p = 0.92
            elif sentiment == 'negative':
                temperature = 0.75  # Moderate creativity for negative
                top_p = 0.9
            else:  # neutral
                temperature = 0.7  # More controlled for neutral
                top_p = 0.85
            
            # Generate text using the pipeline
            result = self.generator(
                enhanced_prompt,
                max_length=60,    # shorter response
                min_length=20,    
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetition
                repetition_penalty=1.2
            )
            
            # Extract and clean the generated text
            generated = result[0]['generated_text']
            
            # Remove the prompt prefix and extract the continuation
            if "Story: " in generated:
                # Find where our actual story continuation starts
                story_parts = generated.split("Story: ", 1)
                if len(story_parts) > 1:
                    generated = story_parts[1]
            
            # Remove the original prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            # Clean up the continuation to remove the sentiment starter if it's there
            for template in ["The atmosphere was", "Everyone shared", "This was", "The feeling", 
                           "Success felt", "The frustration", "Nothing seemed", "The situation",
                           "It was", "The day", "The routine"]:
                if generated.startswith(template):
                    # Keep the sentiment-aligned continuation
                    break
            else:
                # If no template found, try to extract meaningful continuation
                sentences = generated.split('.')
                if len(sentences) > 1:
                    # Take sentences after the first one if it's too similar to input
                    generated = '. '.join(sentences[1:]).strip()
            
            # Ensure we have meaningful content
            if len(generated.strip()) < 20:
                return self._generate_contextual_fallback(prompt, sentiment)
            
            # Ensure proper ending
            if generated and not generated[-1] in '.!?':
                generated += '.'
            
            return generated
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_contextual_fallback(prompt, sentiment)
    
    def _generate_contextual_fallback(self, prompt, sentiment):
        """
        Generate contextually relevant fallback text based on common themes
        
        Args:
            prompt (str): User input prompt
            sentiment (str): Detected sentiment
            
        Returns:
            str: Contextually relevant template-based generated text
        """
        # Analyze prompt for context clues
        prompt_lower = prompt.lower()
        
        # Define context-aware responses
        if sentiment == 'positive':
            if any(word in prompt_lower for word in ['promot', 'success', 'achiev', 'win', 'won']):
                return "Excitement filled the room as congratulations poured in. All the hard work had finally paid off, and the future looked brighter than ever."
            elif any(word in prompt_lower for word in ['love', 'happy', 'joy', 'excit', 'great']):
                return "The joy was contagious, spreading to everyone nearby. It was one of those perfect moments that would be remembered for years to come."
            elif any(word in prompt_lower for word in ['birth', 'baby', 'wedding', 'marry']):
                return "Tears of joy flowed freely as everyone celebrated together. The happiness in the air was palpable, creating memories that would last a lifetime."
            else:
                return "The positive energy was infectious. Everything seemed to fall perfectly into place, creating a sense of accomplishment and satisfaction."
        
        elif sentiment == 'negative':
            if any(word in prompt_lower for word in ['traffic', 'late', 'delay', 'stuck']):
                return "The endless waiting and frustration built up with every passing minute. It was one of those days when nothing seemed to go right."
            elif any(word in prompt_lower for word in ['fail', 'lost', 'broke', 'bad', 'terrible']):
                return "Disappointment washed over like a wave. The setback felt overwhelming, making it hard to see any silver lining."
            elif any(word in prompt_lower for word in ['hate', 'angry', 'frustrat', 'annoy']):
                return "The irritation was almost unbearable. Every little thing seemed to compound the frustration, creating a spiral of negativity."
            else:
                return "The situation felt increasingly difficult to handle. One problem led to another, creating a cascade of complications."
        
        else:  # neutral
            if any(word in prompt_lower for word in ['coffee', 'tea', 'breakfast', 'lunch', 'dinner']):
                return "It was a calm and ordinary moment — just another part of the usual routine. The familiar ritual provided a sense of normalcy."
            elif any(word in prompt_lower for word in ['work', 'office', 'meeting', 'email']):
                return "The day proceeded with its typical rhythm. Tasks were completed methodically, one after another, without any particular urgency."
            elif any(word in prompt_lower for word in ['walk', 'went', 'saw', 'did']):
                return "Nothing particularly noteworthy followed. It was simply another ordinary experience in the flow of daily life."
            else:
                return "Things continued in their usual pattern. Neither particularly good nor bad, just another regular moment passing by."
    
    def _generate_fallback(self, prompt, sentiment):
        """
        Legacy fallback method - redirects to contextual fallback
        
        Args:
            prompt (str): User input prompt
            sentiment (str): Detected sentiment
            
        Returns:
            str: Template-based generated text
        """
        return self._generate_contextual_fallback(prompt, sentiment)