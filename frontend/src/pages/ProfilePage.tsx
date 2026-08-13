import { UserProfile, useUser } from '@clerk/react';
import { SurfaceCard } from '../components/ui/PageShell';

export function ProfilePage(){const {user}=useUser();return <section className="page"><SurfaceCard className="page-card"><header><h1>Profile</h1><p>{user?.fullName ?? user?.primaryEmailAddress?.emailAddress}</p></header><UserProfile routing="hash"/></SurfaceCard></section>}
