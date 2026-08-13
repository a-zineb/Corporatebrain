import { Show, SignInButton, SignUpButton, UserProfile, useUser } from '@clerk/react';
import { SurfaceCard } from '../components/ui/PageShell';

export function ProfilePage(){const {user}=useUser();return <section className="page"><SurfaceCard className="page-card"><header><h1>Profile</h1><p>{user?.fullName ?? user?.primaryEmailAddress?.emailAddress ?? 'Sign in to sync a private profile and history.'}</p></header><Show when="signed-in"><UserProfile routing="hash"/></Show><Show when="signed-out"><div><SignInButton mode="modal"><button className="button">Sign in</button></SignInButton><SignUpButton mode="modal"><button className="button secondary">Create account</button></SignUpButton></div></Show></SurfaceCard></section>}
