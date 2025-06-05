class PhonemeConverter:
    def __init__(self, language='english'):
        self.language = language
        self.phoneme_map = self._load_phoneme_map()
        
    def _load_phoneme_map(self):
        """Load phoneme mapping from file or use default"""
        try:
            with open(f'data/phoneme_lists/{self.language}.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to basic phoneme map
            return self._get_default_phoneme_map()
    
    def _get_default_phoneme_map(self):
        """Return a simple default phoneme map for English or Mandarin"""
        if self.language == 'english':
            return {
                'a': 'AE',
                'e': 'EH',
                'i': 'IH',
                'o': 'AA',
                'u': 'UH',
                'th': 'TH',
                'sh': 'SH',
                'ch': 'CH',
            }
        elif self.language == 'mandarin':
            return {
                'zh': 'ZH',
                'ch': 'CH',
                'sh': 'SH',
                'r': 'R',
                'z': 'Z',
                'c': 'C',
                's': 'S',
                'y': 'Y',
                'w': 'W',
            }
        return {}