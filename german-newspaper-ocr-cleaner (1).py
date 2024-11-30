###############################################################################
# This script defines a cleaner function for newspaper fulltexts              #
###############################################################################

# last major updates November 2024
# contact fredstroessi@gmail.com

import pandas as pd
import re
from difflib import get_close_matches
from collections import defaultdict

class HistoricalGermanOCRCleaner:
    def __init__(self):
        # Enlarge dictionary for 1848-era texts
        self.common_words = {
            # Original words plus historical variants
            'Gewerbsbezirke', 'eingeteilt', 'ständischen', 'König', 'Provinzial',
            'Begutachtung', 'gesammt', 'gewiss', 'Wichtigkeit', 'Ministerium',
            'Bureaukratie', 'Landtag', 'Nationalversammlung', 'Deputirten',
            'Versammlung', 'Regierung', 'Verfassung', 'Freiheit', 'Vaterland', 'Industrie',

            'Interessen', 'Anordnungen', 'Ernennung', 'verhindern', 'Macht', 'haben',
            'Presse', 'nötig', 'nötigen', 'welche', 'jene', 'durch', 'über',
            'Regierung', 'Verwaltung', 'Beamten', 'Versammlung', 'Gesetz',
            
            # Common German articles and prepositions
            'von', 'der', 'die', 'das', 'und', 'für', 'mit', 'sich', 'den', 'zu',
            'ein', 'eine', 'einen', 'einem', 'einer', 'eines', 'in', 'im', 'an',
            'auf', 'ist', 'hat', 'wird', 'sind', 'haben', 'werden'
        }

        # Historical character substitutions specific to 1848 printing
        self.char_substitutions = {
            #'x01': 'n',  # Common error in "Interessen"
            #'l\x01': 'In',
            'ſ': 's',  # Long s
            '=':'-',
            'æ': 'ae',
            'œ': 'oe',
            'â': 'a',
            'ê': 'e',
            'î': 'i',
            'ô': 'o',
            'û': 'u',
            'ÿ': 'y',
            '»': '"',
            '«': '"',
            '„': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '‚': ',',
            '—': '-',
            '–': '-',
        }

        # Enhanced OCR error patterns for 1848 German texts
        self.error_patterns = {
            # Common line break errors
            r'(\w+)-\s*\n\s*(\w+)': r'\1\2',  # Hyphenation fixes
            r'(\w+)\s+\n\s+(\w+)': r'\1 \2',  # Wrong line breaks
            
            # Historical typography fixes
            r'ſ{2}': 'ss',  # Double long s
            r'ſs': 'ss',    # Long s + regular s
            r'ſi': 'si',    # Long s + i
            r'ſt': 'st',    # Long s + t
            r'ſc': 'sc',    # Long s + c
            
            # Common OCR confusions in Fraktur
            r'cb': 'ch',
            r'tb': 'th',
            r'ii': 'n',
            r'rn': 'm',
            r'vv': 'w',
            r'nn': 'rm',
            r'il': 'd',
            #r'I([bcdefghjklmnpqrstuvwxyz])': 'l\1',  # Capital I confused with lowercase l
            
            # Umlaut fixes including historical variants
            r'aͤ': 'ä',
            r'oͤ': 'ö',
            r'uͤ': 'ü',
            r'ae': 'ä',
            r'oe': 'ö',
            #r'ue': 'ü',
            
            # Historical abbreviations
            r'Nr\.': 'Nummer',
            r'Nro\.': 'Nummer',
            r'Hr\.': 'Herr',
            r'Se\.': 'Seine',
            r'Sr\.': 'Seiner',
            r'Majt\.': 'Majestät',
            r'resp\.': 'respektive',
            r'Sr\.\s*Maj\.': 'Seine Majestät',
            r'Pfr\.': 'Pfarrer',
            r'Personenz\.': 'Personenzahl',
            
            # Cleanup patterns
            r'\s+': ' ',           # Multiple spaces
            r'\.+': '.',           # Multiple periods
            r'\,+': ',',           # Multiple commas
            r'[¬\|]': '',         # Remove artifact characters
            r'[\']': '', 
            r'[\(\)\[\]\{\}]': '', # Remove brackets
            r'[„""]': '"',        # Normalize quotes
        }

        # Build word parts dictionary for fuzzy matching
        self.word_parts = defaultdict(set)
        for word in self.common_words:
            for i in range(len(word)-2):
                self.word_parts[word[i:i+3]].add(word)

    def _apply_char_substitutions(self, text):
        """Apply character-level substitutions"""
        for old, new in self.char_substitutions.items():
            text = text.replace(old, new)
        return text

    def _fix_line_continuations(self, text):
        """Handle various types of line continuations and breaks"""
        # Remove soft hyphens at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', lambda m: self._join_hyphenated_words(m.group(1), m.group(2)), text)
        
        # Fix spaces around line breaks
        text = re.sub(r'\s*\n\s*', ' ', text)
        
        return text

    def _join_hyphenated_words(self, part1, part2):
        """Smart joining of hyphenated words with historical context"""
        joined = part1 + part2
        
        # Check if joined word exists in common words
        if joined in self.common_words:
            return joined
            
        # Check for common word parts
        for i in range(len(joined)-2):
            if joined[i:i+3] in self.word_parts:
                return joined
                
        # If uncertain, maintain hyphenation
        return f"{part1}-{part2}"

    def _normalize_spacing(self, text):
        """Normalize spacing around punctuation and special characters"""
        # Fix spaces around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
        
        # Normalize quote spacing
        text = re.sub(r'\s*"\s*', '" ', text)
        
        return text.strip()

    def clean_text(self, text):
        if pd.isna(text):
            return text
            
        # Convert to string if not already
        text = str(text)
        
        # Initial cleanup
        text = text.strip()
        
        # Apply character substitutions
        text = self._apply_char_substitutions(text)
        
        # Fix line continuations
        text = self._fix_line_continuations(text)
        
        # Apply error patterns
        for pattern, replacement in self.error_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize spacing
        text = self._normalize_spacing(text)
        
        return text

    def process_file(self, input_file, output_file, text_column=None):
        """Process a file with historical German text"""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                if input_file.endswith('.csv'):
                    df = pd.read_csv(input_file, encoding=encoding)
                    if text_column:
                        df[text_column + '_cleaned'] = df[text_column].apply(self.clean_text)
                    else:
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                df[col + '_cleaned'] = df[col].apply(self.clean_text)
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    return df
                else:
                    with open(input_file, 'r', encoding=encoding) as f:
                        text = f.read()
                    cleaned_text = self.clean_text(text)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    return cleaned_text
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")

# Example usage
if __name__ == "__main__":
    cleaner = HistoricalGermanOCRCleaner()
    
    # Test with example text
    test_text = """
    Die Bureaukratie alten Schlages , die aus den Zeiten Friedrich
    Wilhelm des Drit- n M ten her daran gewöhnt war , daſs die
    höchſten Ehrenſtellen ſich im Staate nur auf der langen Leiter
    der Unterbedienung beſt« und Mittelſtellen erklommen werden
    durften.
    """
    
    cleaned = cleaner.clean_text(test_text)
    print("Original:", test_text)
    print("Cleaned:", cleaned)