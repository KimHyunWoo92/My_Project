import firebase_admin
from firebase_admin import credentials, firestore

class GCFS:
    # _initialized = False
    # KEY_PATH='/Users/hyeok/Desktop/Development/Python/deepbuger/src/google_cloud/config/fake-24f85-firebase-adminsdk-siobk-4087a6ab01.json'

    # @classmethod
    # def initialize_app(cls):
    #     if not cls._initialized:
    #         cred = credentials.Certificate(KEY_PATH)
    #         initialize_app(cred)
    #         cls._initialized = True

    def __init__(self):
        self.KEY_PATH='src/google_cloud/config/fake-24f85-firebase-adminsdk-siobk-4087a6ab01.json'
        self.cred = credentials.Certificate(self.KEY_PATH)
        self.collection_name='predicts'
        self.db = None  # 초기화를 나중에 수행하기 위해 None으로 초기화
        

    def initialize_db(self):
        # print(self.db)
        if self.db is None:  # 이미 초기화되었는지 확인
            firebase_admin.initialize_app(self.cred)
            self.db = firestore.client()
        


    def select_data(self,db):
        pass

    def new_doc_id(self):
        id=0
        
        predicts_ref = self.db.collection(self.collection_name)
        docs = predicts_ref.stream()

        for doc  in docs:
            if int(doc.id) > id:
                id = int(doc.id)
        return id + 1

    def update_data(self,db):
        pass

    def insert_data(self,input_url,output_url,predict=0.00):
        self.initialize_db()
        if predict >= 50.00:
            result = True
        else:
            result = False
        id=self.new_doc_id()
        doc_ref = self.db.collection(self.collection_name).document(str(id))
        print(f"create new document id : {id}")
        doc_ref.set({
            u'input_url':input_url,
            u'output_url':output_url,
            u'predict':predict,
            u'result':result,
        })

    # filepath='/Users/hyeok/Desktop/Development/Python/google_cloud/video/jennie_df.avi'
    # predict=69
    # result=True

    # insert_data(filepath,predict,result)

    # print(new_doc_id())