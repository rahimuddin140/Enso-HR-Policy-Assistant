import express from 'express';
import fs from 'fs';
import path from 'path';
import multer from 'multer';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { RAGEngine } from './src/server/ragEngine.js';

dotenv.config();

const EVALUATION_QUESTIONS = [
  'How many days of Privilege Leave (PL) does a permanent employee get per year?',
  'What is the probation period for employees up to Senior Manager?',
  "What is the company's health-insurance premium amount?",
  'What is the current GST rate in India?',
  'How many days of maternity leave are given?',
];

const uploadDir = path.resolve('./data/uploads');
fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({
  dest: uploadDir,
  limits: {
    fileSize: 25 * 1024 * 1024, // 25 MB max
  },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype !== 'application/pdf' && !file.originalname.toLowerCase().endsWith('.pdf')) {
      cb(new Error('Invalid file type. Only PDF files (.pdf) are allowed.'));
      return;
    }
    cb(null, true);
  },
});

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '2mb' }));

  const ragEngine = new RAGEngine();
  await ragEngine.ensureInitialized().catch((err) => {
    console.warn('Initial PDF auto-ingestion warning:', err.message);
  });

  app.get('/api/status', (_req, res) => {
    try {
      res.json(ragEngine.getStatus());
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to read index status.' });
    }
  });

  app.post('/api/ask', async (req, res) => {
    try {
      const question = String(req.body?.question || '').trim();
      if (!question) {
        res.status(400).json({
          error: 'Please enter a question before submitting.',
          answer: 'Not found',
          found: false,
          sources: [],
          message: 'Question input was empty.',
        });
        return;
      }
      const contract = await ragEngine.answerQuestion(question);
      res.json(contract);
    } catch (err: any) {
      res.status(400).json({
        error: err.message || 'Failed to process question.',
        answer: 'Not found',
        found: false,
        sources: [],
        message: err.message || 'Retrieval error.',
      });
    }
  });

  app.post('/api/ingest', async (_req, res) => {
    try {
      const status = await ragEngine.ingestPdf();
      res.json({
        success: true,
        message: `Successfully indexed ${status.chunkCount} chunks across ${status.nonEmptyPages}/${status.pageCount} pages from ${status.pdfFilename}.`,
        status,
      });
    } catch (err: any) {
      res.status(400).json({
        success: false,
        error: err.message || 'Failed to ingest PDF.',
      });
    }
  });

  app.post('/api/upload-pdf', (req, res) => {
    upload.single('pdf')(req, res, async (uploadErr: any) => {
      if (uploadErr) {
        res.status(400).json({
          success: false,
          error: uploadErr.message || 'File upload failed.',
        });
        return;
      }

      const file = req.file;
      if (!file) {
        res.status(400).json({
          success: false,
          error: 'No PDF file was provided in the upload request.',
        });
        return;
      }

      try {
        const targetPdf = path.resolve('./data/hr_manual.pdf');
        fs.copyFileSync(file.path, targetPdf);
        fs.unlinkSync(file.path);
        const safeName = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
        const status = await ragEngine.ingestPdf(targetPdf, safeName || 'hr_manual.pdf');
        res.json({
          success: true,
          message: `Uploaded and indexed '${safeName}' (${status.pageCount} pages, ${status.chunkCount} chunks).`,
          status,
        });
      } catch (err: any) {
        if (file.path && fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
        res.status(400).json({
          success: false,
          error: err.message || 'Uploaded PDF could not be parsed.',
        });
      }
    });
  });

  app.get('/api/evaluate', (_req, res) => {
    const reportPath = path.resolve('./data/evaluation_report.json');
    if (fs.existsSync(reportPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
        res.json({ results: data });
        return;
      } catch {
        // fall through
      }
    }
    res.json({ results: [] });
  });

  app.post('/api/evaluate', async (_req, res) => {
    try {
      const results = [];
      for (let i = 0; i < EVALUATION_QUESTIONS.length; i++) {
        const q = EVALUATION_QUESTIONS[i];
        try {
          const ans = await ragEngine.answerQuestion(q);
          results.push({
            id: i + 1,
            question: q,
            answer: ans.answer,
            found: ans.found,
            sources: ans.sources,
            supported_by_evidence: Boolean(ans.found && ans.sources.length > 0),
            message: ans.message,
            error: null,
          });
        } catch (err: any) {
          results.push({
            id: i + 1,
            question: q,
            answer: 'Not found',
            found: false,
            sources: [],
            supported_by_evidence: false,
            message: '',
            error: err.message || 'Evaluation failed',
          });
        }
      }

      const reportPath = path.resolve('./data/evaluation_report.json');
      fs.mkdirSync(path.dirname(reportPath), { recursive: true });
      fs.writeFileSync(reportPath, JSON.stringify(results, null, 2), 'utf-8');
      res.json({ results });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to run evaluation.' });
    }
  });

  app.get('/api/chunks', (req, res) => {
    try {
      const query = req.query.q ? String(req.query.q) : undefined;
      const page = req.query.page ? parseInt(String(req.query.page), 10) : undefined;
      const chunks = ragEngine.getChunks(query, page);
      res.json({
        total: chunks.length,
        chunks: chunks.slice(0, 80),
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to query chunks.' });
    }
  });

  app.get('/api/project-files', (_req, res) => {
    const filePaths = [
      'README.md',
      'requirements.txt',
      '.env.example',
      'app.py',
      'src/hr_assistant/config.py',
      'src/hr_assistant/schemas.py',
      'src/hr_assistant/ingestion.py',
      'src/hr_assistant/embeddings.py',
      'src/hr_assistant/vector_store.py',
      'src/hr_assistant/retrieval.py',
      'src/hr_assistant/answer_service.py',
      'scripts/ingest.py',
      'scripts/evaluate_questions.py',
      'tests/test_ingestion.py',
      'tests/test_retrieval.py',
      'tests/test_answer_service.py',
      'tests/test_citations.py',
      'tests/test_ui_smoke.py',
    ];
    const files: { path: string; content: string }[] = [];
    for (const relPath of filePaths) {
      const abs = path.resolve(relPath);
      if (fs.existsSync(abs)) {
        files.push({
          path: relPath,
          content: fs.readFileSync(abs, 'utf-8'),
        });
      }
    }
    res.json({ files });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Enso HR Policy Assistant server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
