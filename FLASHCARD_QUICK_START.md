# 🎓 Flashcard System - Quick Start Guide

## ✅ Server is Running!

The backend is working correctly. The `/profiles` endpoint returns `[]` (empty array) because there are no completed research sessions yet.

---

## 📝 Step-by-Step Instructions

### **1. Open the UI**

Go to: http://localhost:8001

### **2. Upload Papers**

- Click **"Upload PDFs"** button
- Select 2-3 research papers (PDF files)
- Wait for upload to complete
- You should see: "✅ X papers uploaded successfully!"

### **3. Run Deep Research**

- Type a research question in the input box
- Click the **"Deep Research"** toggle (make sure it's ON/blue)
- Click **Send** or press Enter
- Wait for the pipeline to complete (this takes 2-5 minutes)
- You'll see progress messages and a final report

### **4. View Flashcards**

- After Deep Research completes, click the **"Flashcards"** button in the header
  - It's the cyan/teal button with a card stack icon
  - Located next to the "Graph" button
- The flashcard modal will open showing:
  - 5 flashcards per paper
  - Click any card to flip and see the answer
  - Stats showing Papers / Flashcards / Reviewed count

---

## 🔍 Troubleshooting

### **"No Flashcards Available" message**

This means one of these:

1. ❌ **No Deep Research completed yet**
   - Solution: Run a Deep Research query first

2. ❌ **Deep Research failed**
   - Check terminal for error messages
   - Look for "Pipeline completed" in logs

3. ❌ **Wrong mode used**
   - Make sure "Deep Research" toggle is ON (blue)
   - Regular chat mode doesn't generate flashcards

### **Check Backend Logs**

After Deep Research completes, you should see:

```
✅ Profile Builder: 2 structured profiles created
Generated 5 flashcards for paper 'Title' (ID: abc12345)
✅ Flashcards integrated into profile for 'Title'
```

When you click Flashcards button:

```
Profiles request - session_id: auto, total sessions: 1
Found session with profiles: abc12345 (status: completed)
Profiles endpoint: returning 2 profiles with 10 total flashcards
```

### **Check Browser Console**

Open DevTools (F12) → Console tab. You should see:

```javascript
Flashcard data received: {
  profileCount: 2,
  profiles: [...],
  hasFlashcards: true
}
```

---

## ✅ Expected Behavior

### **Before Deep Research:**
- Flashcards button is visible but modal shows "No Flashcards Available"
- `/profiles` endpoint returns `[]`

### **After Deep Research:**
- Flashcards button opens modal with cards
- Each paper has exactly 5 flashcards:
  1. Research Objective
  2. Key Finding I
  3. Key Finding II
  4. Methodology
  5. Critical Takeaway

### **Flashcard Features:**
- ✅ 3D flip animation (click to reveal answer)
- ✅ Technical Blueprint theme (dark navy + cyan)
- ✅ Stats tracking (Papers / Flashcards / Reviewed)
- ✅ Grouped by paper with paper titles
- ✅ Responsive grid layout

---

## 🎯 Quick Test

Run this in PowerShell to verify the server is working:

```powershell
# Test health
curl -UseBasicParsing http://localhost:8001/health

# Test profiles (should return [] if no sessions)
curl -UseBasicParsing http://localhost:8001/profiles
```

Both should return HTTP 200 OK.

---

## 📞 Still Not Working?

If flashcards still don't appear after completing Deep Research:

1. **Check terminal logs** - look for "Generated 5 flashcards"
2. **Check browser console** - look for errors or `hasFlashcards: false`
3. **Try refreshing the page** - sometimes the UI needs a refresh
4. **Clear browser cache** - Ctrl+Shift+Delete
5. **Restart backend** - Stop and start uvicorn again

---

**Last Updated**: 2026-05-14  
**Status**: Server Running ✅ | Waiting for Deep Research
