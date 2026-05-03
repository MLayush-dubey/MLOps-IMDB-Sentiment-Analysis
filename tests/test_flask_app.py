import unittest 
from flask_app.app import app 

class FlaskAppTests(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        cls.client = app.test_client()   #flask tool which creates a dummy browser to send requests

    def test_home_page(self):
        response = self.client.get("/")  #simulates user visiting the root URL
        self.assertEqual(response.status_code, 200)  
        self.assertIn(b"<title>IMDB Sentiment Analysis</title>", response.data)

    
    def test_predict_page(self):
        response = self.client.post("/predict", data = dict(text = "This is a great movie"))  #simulates submitting a form
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,   #checks upon hitting predict, do we get positive or negative
            "Response should contain either Positive or Negative"
        )


if __name__ == "__main__":
    unittest.main()