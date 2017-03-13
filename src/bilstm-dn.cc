#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <random>

using namespace std;
using namespace cnn;

//float pdrop = 0.5;
unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool eval = false;
cnn::Dict d;
cnn::Dict td;
int kNONE;
int kSOS;
int kEOS;

// default epochs
unsigned MAX_EPOCHS = 10;

// use the universal tagset
const string TAG_SET[] = {"VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."};

template <class Builder>
struct RNNJointModel
{
  LookupParameters *p_w;
  Parameters *p_l2th;
  Parameters *p_r2th;
  Parameters *p_thbias;

  Parameters *p_th2t;
  Parameters *p_tbias;
  Builder l2rbuilder;
  Builder r2lbuilder;

  // noise layer
  Parameters *p_nl;

  explicit RNNJointModel(Model &model) : l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
                 r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model)
  {
    p_w = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
    p_l2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_r2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

    p_th2t = model.add_parameters({TAG_SIZE, TAG_HIDDEN_DIM});
    p_tbias = model.add_parameters({TAG_SIZE});

    // noise layer
    p_nl = model.add_parameters({TAG_SIZE, TAG_SIZE});
  }

  Expression BuildTaggingGraphWithNoise(const vector<int> &sent, const vector<int> &tags, ComputationGraph &cg, double *cor = 0, unsigned *ntagged = 0, unsigned isNoNoise = 0)
  {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    r2lbuilder.start_new_sequence();
    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias);
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = lookup(cg, p_w, sent[t]);
      if (!eval)
      {
        i_words[t] = noise(i_words[t], 0.1);
      }
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(lookup(cg, p_w, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    for (unsigned t = 0; t < slen; ++t)
    {
      if (tags[t] != kNONE)
      {
        if (ntagged)
          (*ntagged)++;
        Expression i_th = tanh(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
        if (cor)
        {
          vector<float> dist = as_vector(cg.incremental_forward());
          double best = -9e99;
          int besti = -1;
          for (int i = 0; i < dist.size(); ++i)
          {
            if (dist[i] > best)
            {
              best = dist[i];
              besti = i;
            }
          }
          if (tags[t] == besti)
            (*cor)++;
        }
        // different objectives
        if (isNoNoise == 1)
        {
          //Expression i_a = const_parameter(cg, p_nl); //const but no use
          Expression i_err = pickneglogsoftmax(i_t, tags[t]);
          errs.push_back(i_err);
        }
        else
        {
          Expression i_a = parameter(cg, p_nl);
          Expression i_nl = i_a * i_t;
          Expression i_nl_err = pickneglogsoftmax(i_nl, tags[t]);
          errs.push_back(i_nl_err);
        }
      }
    }
    return sum(errs);
  }

  // predict the tags of an inpute sentence
  vector<string> PredictSequentTags(const vector<int> &sent, const vector<int> &tags, ComputationGraph &cg, double *cor = 0, unsigned *ntagged = 0)
  {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg); // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg); // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();
    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias);
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    vector<string> preds;

    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = lookup(cg, p_w, sent[t]);
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }
    // read sequence from right to left
    r2lbuilder.add_input(lookup(cg, p_w, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    for (unsigned t = 0; t < slen; ++t)
    {

      if (ntagged)
        (*ntagged)++;
      Expression i_th = tanh(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));

      Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
      if (cor)
      {
        vector<float> dist = as_vector(cg.incremental_forward());
        double best = -9e99;
        int besti = -1;
        for (int i = 0; i < dist.size(); ++i)
        {
          if (dist[i] > best)
          {
            best = dist[i];
            besti = i;
          }
        }
        if (tags[t] == besti)
          (*cor)++;
      }
      double best = 9e+99;
      string ptag;
      for (const string &tag : TAG_SET)
      {
        Expression i_err_t = pickneglogsoftmax(i_t, td.Convert(tag));
        double error = as_scalar(i_err_t.value());
        if (error < best)
        {
          best = error;
          ptag = tag;
        }
      }
      preds.push_back(ptag);
    }
    return preds;
  }
};

int main(int argc, char **argv)
{
  cnn::Initialize(argc, argv);
  kNONE = td.Convert("*");
  // use universal tagset
  for (const string &tag : TAG_SET)
  {
    td.Convert(tag);
  }
  td.Freeze(); // no new tag types allowed
  TAG_SIZE = td.size();

  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<pair<vector<int>, vector<int>>> training, dev, test;
  vector<unsigned> data_type;

  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading supervision data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while (getline(in, line))
    {
      ++tlc;
      //read both the words and tags
      vector<int> x, y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      if (x.size() == 0)
      {
        cerr << line << endl;
        abort();
      }
      training.push_back(make_pair(x, y));
      // no noise
      data_type.push_back(1);
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }

  tlc = 0;
  ttoks = 0;
  cerr << "Reading projection data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while (getline(in, line))
    {
      ++tlc;
      //read both the words and tags
      vector<int> x, y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      if (x.size() == 0)
      {
        cerr << line << endl;
        abort();
      }
      training.push_back(make_pair(x, y));
      // exist noise
      data_type.push_back(0);
      ttoks += x.size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }

  d.Freeze(); // no new word types allowed
  d.SetUnk("<UNK>");
  VOCAB_SIZE = d.size();
  assert(training.size() == data_type.size());

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[3] << "...\n";
  {
    ifstream in(argv[3]);
    assert(in);
    while (getline(in, line))
    {
      ++dlc;
      vector<int> x, y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      dev.push_back(make_pair(x, y));
      dtoks += x.size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }

  int telc = 0;
  int tetoks = 0;
  cerr << "Reading test data from " << argv[4] << "...\n";
  {
    ifstream in(argv[4]);
    assert(in);
    while (getline(in, line))
    {
      ++telc;
      vector<int> x, y;
      ReadSentencePair(line, &x, &d, &y, &td);
      assert(x.size() == y.size());
      test.push_back(make_pair(x, y));
      tetoks += x.size();
    }
    cerr << telc << " lines, " << tetoks << " tokens\n";
  }

  MAX_EPOCHS = stoi(argv[5]);

  double best = 9e+99;

  Model model;
  bool use_momentum = true;
  Trainer *sgd = nullptr;
  float lambda = 1e-3;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model, lambda);
  else
    sgd = new SimpleSGDTrainer(&model);

  RNNJointModel<LSTMBuilder> lm(model);

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i)
    order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  int epochs = 0;
  while (1)
  {

    if (si == training.size())
    {
      epochs++;
    }

    if (epochs > MAX_EPOCHS)
    {
      break;
    }

    Timer iteration("completed in");
    double loss = 0;
    unsigned ttags = 0;
    double correct = 0;

    //train supervsion model at first
    if (si == 0 || si == training.size())
    {
      for (unsigned i = 0; i < training.size(); ++i)
      {
        if (data_type[i] == 0)
          continue;

        // build graph for this instance
        ComputationGraph cg;
        auto &sent = training[i];
        unsigned dtype = data_type[i];

        lm.BuildTaggingGraphWithNoise(sent.first, sent.second, cg, &correct, &ttags, dtype);
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd->update();
      }
    }

    for (unsigned i = 0; i < report_every_i; ++i)
    {
      if (si == training.size())
      {
        si = 0;
        if (first)
        {
          first = false;
        }
        else
        {
          sgd->update_epoch();
        }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }

      // build graph for this instance
      ComputationGraph cg;
      auto &sent = training[order[si]];
      unsigned dtype = data_type[order[si]];

      ++si;
      lm.BuildTaggingGraphWithNoise(sent.first, sent.second, cg, &correct, &ttags, dtype);
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";

    // show score on dev data
    report++;
    if (report % dev_every_i_reports == 0)
    {
      double dloss = 0;
      unsigned dtags = 0;
      double dcorr = 0;
      eval = true;
      for (auto &sent : dev)
      {
        ComputationGraph cg;
        lm.BuildTaggingGraphWithNoise(sent.first, sent.second, cg, &dcorr, &dtags, 1);
        dloss += as_scalar(cg.forward());
      }

      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';

      eval = false;
      if (dloss < best)
      {
        best = dloss;
        unsigned postive = 0;
        unsigned tokens = 0;

        unsigned dtags = 0;
        double dcorr = 0;

        for (auto &sent : test)
        {
          ComputationGraph cg;
          vector<string> preds = lm.PredictSequentTags(sent.first, sent.second, cg, &dcorr, &dtags);

          for (unsigned ti = 1; ti < sent.first.size() - 1; ti++)
          {
            if (td.Convert(preds[ti - 1]) == sent.second[ti])
            {
              postive++;
            }
            tokens++;
          }
        }

        cerr << "\n***Best [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';

        double accuracy = postive * 1.0 / tokens;
        cerr << "\n >> Accuracy: " << accuracy << "\n";
      }
    }
  }
  delete sgd;

  cout << ":) Good luck :)" << endl;
}
