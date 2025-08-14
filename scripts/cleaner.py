import shutil
import language_tool_python


class TextCleaner():

    def __init__(self,
                 txt_lang:str):
        
        if not self.is_java_installed():
            self.tool = None # LanguageToolPublicAPI does not work
            print("Please install java to use language-tool-python")
        
        elif txt_lang.lower() == "jp":
            print('''Japanese Language is not fully supported in language_tool_python.\n
                          \n''')
            
            answer = input("Proceed anyway? (Y/N)").strip().lower()
            if answer == 'y':
                self.tool = language_tool_python.LanguageTool(language=txt_lang)
            else:
                self.tool = None
              
        else:
            self.tool = language_tool_python.LanguageTool(language=txt_lang)
    
    def _language_tool(self,text:str):
        return language_tool_python.utils.correct(
            text,
            self.tool.check(text))
    
    def clean_up(self,text:str|list[str]) -> str|list[str]:
        if isinstance(text,list):
            return list(map(self._language_tool,text))
        elif isinstance(text,str):
            return self._language_tool(text)
        else:
            return text #maybe a generator ?
    
    @staticmethod
    def is_java_installed() -> bool:
        return shutil.which("java") is not None
        
    
    
    