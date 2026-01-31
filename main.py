from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import hashlib


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# CONSTANTS
MAX_ITEMS_PER_COUPLE = 250


# ============ MODELS ============

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class CoupleRoomCreate(BaseModel):
    coupleCode: str

class CoupleRoom(BaseModel):
    coupleCode: str
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PhotoCreate(BaseModel):
    coupleCode: str
    imageData: str
    caption: str = ""
    uploadedBy: str

class Photo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    coupleCode: str
    imageData: str
    caption: str
    uploadedBy: str
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SecretNoteCreate(BaseModel):
    coupleCode: str
    title: str
    content: str
    passcode: str = "love"
    createdBy: str
    lockedUntil: Optional[str] = None

class SecretNote(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    coupleCode: str
    title: str
    content: str
    passcodeHash: str
    createdBy: str
    lockedUntil: Optional[str] = None
    isUnlocked: bool = False
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TriggerPullSessionCreate(BaseModel):
    coupleCode: str
    createdBy: str

class TriggerPullResponse(BaseModel):
    sessionId: str
    respondedBy: str
    response: str

class TriggerPullSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    coupleCode: str
    promptId: int
    promptText: str
    createdBy: str
    prabhResponse: Optional[str] = None
    sehajResponse: Optional[str] = None
    revealed: bool = False
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PresenceUpdate(BaseModel):
    coupleCode: str
    user: str
    action: str
    message: Optional[str] = None

class PresenceState(BaseModel):
    coupleCode: str
    lastVisitAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lastActionAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lastMessage: Optional[str] = None
    lastKissAt: Optional[datetime] = None
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UnlockNoteRequest(BaseModel):
    passcode: str


# ============ HELPER FUNCTIONS ============

def serialize_doc(doc):
    if doc is None:
        return None
    result = {}
    for key, value in doc.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result

def serialize_docs(docs):
    return [serialize_doc(doc) for doc in docs]

async def enforce_cap_limit(collection_name: str, couple_code: str):
    count = await db[collection_name].count_documents({"coupleCode": couple_code})
    if count > MAX_ITEMS_PER_COUPLE:
        excess = count - MAX_ITEMS_PER_COUPLE
        oldest = await db[collection_name].find({"coupleCode": couple_code}).sort("createdAt", 1).limit(excess).to_list(excess)
        for item in oldest:
            await db[collection_name].delete_one({"id": item["id"]})


# ============ ROUTES ============

@api_router.get("/")
async def root():
    return {"message": "Love App API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.model_dump())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**{k: v for k, v in sc.items() if k != '_id'}) for sc in status_checks]

@api_router.post("/couple/verify")
async def verify_couple_code(data: CoupleRoomCreate):
    couple = await db.couple_rooms.find_one({"coupleCode": data.coupleCode})
    if not couple:
        room = CoupleRoom(coupleCode=data.coupleCode)
        await db.couple_rooms.insert_one(room.model_dump())
    return {"success": True, "coupleCode": data.coupleCode}

# SHARED ALBUM
@api_router.get("/album/photos")
async def get_photos(coupleCode: str):
    photos = await db.shared_album.find({"coupleCode": coupleCode}, {"_id": 0}).sort("createdAt", -1).to_list(MAX_ITEMS_PER_COUPLE)
    return photos

@api_router.post("/album/photos")
async def add_photo(photo: PhotoCreate):
    photo_obj = Photo(**photo.model_dump())
    await db.shared_album.insert_one(photo_obj.model_dump())
    await enforce_cap_limit("shared_album", photo.coupleCode)
    return {k: v for k, v in photo_obj.model_dump().items() if k != '_id'}

@api_router.delete("/album/photos/{photo_id}")
async def delete_photo(photo_id: str):
    result = await db.shared_album.delete_one({"id": photo_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Photo not found")
    return {"success": True}

# LOVE VAULT
@api_router.get("/vault/notes")
async def get_vault_notes(coupleCode: str):
    notes = await db.love_vault.find({"coupleCode": coupleCode}, {"_id": 0}).sort("createdAt", -1).to_list(MAX_ITEMS_PER_COUPLE)
    return notes

@api_router.post("/vault/notes")
async def create_vault_note(note: SecretNoteCreate):
    passcode_hash = hashlib.sha256(note.passcode.encode()).hexdigest()
    note_dict = note.model_dump()
    note_dict.pop('passcode')
    note_dict['passcodeHash'] = passcode_hash
    note_obj = SecretNote(**note_dict)
    await db.love_vault.insert_one(note_obj.model_dump())
    await enforce_cap_limit("love_vault", note.coupleCode)
    return {k: v for k, v in note_obj.model_dump().items() if k != '_id'}

@api_router.post("/vault/notes/{note_id}/unlock")
async def unlock_vault_note(note_id: str, data: UnlockNoteRequest):
    note = await db.love_vault.find_one({"id": note_id}, {"_id": 0})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    passcode_hash = hashlib.sha256(data.passcode.encode()).hexdigest()
    if note["passcodeHash"] != passcode_hash:
        return {"success": False, "error": "Wrong passcode"}
    if note.get("lockedUntil"):
        lock_time = datetime.fromisoformat(note["lockedUntil"].replace('Z', '+00:00'))
        if datetime.now(timezone.utc) < lock_time:
            return {"success": False, "error": "Note is still time-locked"}
    await db.love_vault.update_one({"id": note_id}, {"$set": {"isUnlocked": True, "updatedAt": datetime.now(timezone.utc)}})
    return {"success": True, "content": note.get("content", "")}

@api_router.delete("/vault/notes/{note_id}")
async def delete_vault_note(note_id: str):
    result = await db.love_vault.delete_one({"id": note_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"success": True}

# TRIGGER PULL
TRIGGER_PROMPTS = [
    "What do you love most about me?",
    "What's your favorite memory with me?",
    "What makes you feel loved by me?",
    "What's something I do that makes you smile?",
    "What do you want to do with me this weekend?",
    "What's one thing you want to try together?",
    "What's your favorite thing about our relationship?",
    "When did you first know you loved me?",
    "What's something you want me to know?",
    "What's a compliment you've been saving for me?",
]

@api_router.get("/trigger/sessions")
async def get_trigger_sessions(coupleCode: str):
    sessions = await db.trigger_pull.find({"coupleCode": coupleCode}, {"_id": 0}).sort("createdAt", -1).to_list(MAX_ITEMS_PER_COUPLE)
    return sessions

@api_router.post("/trigger/sessions")
async def create_trigger_session(data: TriggerPullSessionCreate):
    import random
    prompt_id = random.randint(0, len(TRIGGER_PROMPTS) - 1)
    prompt_text = TRIGGER_PROMPTS[prompt_id]
    session = TriggerPullSession(coupleCode=data.coupleCode, promptId=prompt_id, promptText=prompt_text, createdBy=data.createdBy)
    await db.trigger_pull.insert_one(session.model_dump())
    await enforce_cap_limit("trigger_pull", data.coupleCode)
    return {k: v for k, v in session.model_dump().items() if k != '_id'}

@api_router.post("/trigger/sessions/{session_id}/respond")
async def respond_to_trigger(session_id: str, data: TriggerPullResponse):
    session = await db.trigger_pull.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    update_field = "prabhResponse" if data.respondedBy == "prabh" else "sehajResponse"
    update_data = {update_field: data.response}
    other_field = "sehajResponse" if data.respondedBy == "prabh" else "prabhResponse"
    if session.get(other_field):
        update_data["revealed"] = True
    await db.trigger_pull.update_one({"id": session_id}, {"$set": update_data})
    updated = await db.trigger_pull.find_one({"id": session_id}, {"_id": 0})
    return updated

# PRESENCE
@api_router.get("/presence/state")
async def get_presence_state(coupleCode: str):
    state = await db.presence_state.find_one({"coupleCode": coupleCode}, {"_id": 0})
    if not state:
        initial = PresenceState(coupleCode=coupleCode)
        await db.presence_state.insert_one(initial.model_dump())
        return serialize_doc(initial.model_dump())
    return serialize_doc(state)

@api_router.post("/presence/update")
async def update_presence(data: PresenceUpdate):
    update_fields = {"lastActionAt": datetime.now(timezone.utc), "updatedAt": datetime.now(timezone.utc)}
    if data.action == "visit":
        update_fields["lastVisitAt"] = datetime.now(timezone.utc)
    elif data.action == "message" and data.message:
        update_fields["lastMessage"] = data.message
    elif data.action == "kiss":
        update_fields["lastKissAt"] = datetime.now(timezone.utc)
    await db.presence_state.update_one({"coupleCode": data.coupleCode}, {"$set": update_fields}, upsert=True)
    state = await db.presence_state.find_one({"coupleCode": data.coupleCode}, {"_id": 0})
    return serialize_doc(state)


# Include the router
app.include_router(api_router)

# CORS
origins = [
    "https://love-me-sehaj-final-2sup.vercel.app",
    "https://love-me-sehaj-1e3k.vercel.app",
    "http://localhost:3001",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
