import unittest
import torch
from q1_image_captioning.tokenizer import Tokenizer
from q1_image_captioning.collate import pad_sequences, collate_fn
from q1_image_captioning.models import EncoderCNN, DecoderLSTM

class TestQ1(unittest.TestCase):
    def test_tokenizer(self):
        toks = ['A cat on the sofa', 'A dog in the park']
        tk = Tokenizer()
        tk.build_vocab(toks)
        seq = tk.text_to_sequence('A cat on the sofa')
        text = tk.sequence_to_text(seq)
        self.assertIn('cat', text)

    def test_pad(self):
        s = [[1,2,3], [4,5]]
        out, lengths = pad_sequences(s, pad_value=0)
        self.assertEqual(out.shape[1], 3)
        self.assertTrue((lengths==torch.tensor([3,2])).all())

    def test_model_shapes(self):
        enc = EncoderCNN(encoded_dim=512, freeze_backbone=True)
        dec = DecoderLSTM(vocab_size=50, embed_dim=32, decoder_dim=64, encoder_dim=512)
        imgs = torch.rand(2,3,224,224)
        enc_out = enc(imgs)
        # create fake captions
        caps = torch.randint(0,49,(2,10))
        logits, alphas = dec(caps, enc_out)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(alphas.shape[0], 2)

if __name__ == '__main__':
    unittest.main()
